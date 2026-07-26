"""Shrink the tri-modal cascade GLBs to something that can live in a git repo.

The cascade writes ~7 MB per mesh (119k verts / 199k faces + two 2048^2 textures).
42 of those is 280 MB, which is not going in the hub.

The thing that makes this awkward is the UV atlas. TRELLIS bakes one with xatlas, which
splits the mesh at every chart boundary, and a quadric decimator will not collapse
across those boundaries. Decimating in place therefore *concentrates* the seams: asking
for 25k faces gave meshes with 45k vertices (a closed 25k-face mesh wants ~12.5k), one
asset refused to go below 59k faces at all, and the total landed at 34 MB. So instead:

1. Weld the mesh back together by position, throw the atlas away, and run a plain
   quadric edge collapse -- with no seams to protect it reaches the target every time.
2. Re-parametrise the decimated mesh with xatlas and re-bake the baseColor: rasterise
   the new atlas, push every texel back to the closest point on the *original* surface,
   and read the original 2048^2 texture there. Seam bleed is filled by a nearest-valid
   dilation.
3. Fold the metallic/roughness map into scalar factors when it is constant, which it
   nearly always is (TRELLIS emits roughness=1, metallic=0 almost everywhere).
4. Re-encode with KHR_mesh_quantization -- int16 positions behind a node scale, int8
   normals, uint16 UVs, uint16 indices. That halves the vertex buffer, which buys back
   the room to ship smooth NORMALs (trimesh's own exporter drops them, and flat-shaded
   25k-face organics look like origami).

Fidelity is *measured*, not assumed: every output is rendered against its source from
8 viewpoints with nvdiffrast and scored on silhouette IoU, so thin structures that
decimate badly show up in the report instead of quietly shipping.

    python scripts/compress_trimodal_glbs.py [--faces 25000] [--tex 768]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import struct

import numpy as np
import trimesh
from PIL import Image

SRC_DIR = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/trimodal"
OUT_DIR = os.path.join(SRC_DIR, "web")
REPORT = os.path.join(SRC_DIR, "web_compression_report.json")

# per-file overrides for meshes whose silhouette does not survive the default budget
FACE_OVERRIDES: dict[str, int] = {}


# --------------------------------------------------------------------------- mesh

def load_glb(path):
    scene = trimesh.load(path, process=False)
    geoms = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
    assert len(geoms) == 1, f"{path}: expected a single mesh, got {len(geoms)}"
    return geoms[0]


def gltf_uv(mesh):
    """trimesh flips v on glTF import (origin to lower-left) and flips it back on
    export. We write the GLB by hand, so undo the import flip once, here, and work in
    glTF convention everywhere after: v=0 is the *top* row of the texture image."""
    uv = np.asarray(mesh.visual.uv, dtype=np.float64).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def weld(V, F):
    """Merge vertices the atlas duplicated, and drop the faces that go degenerate."""
    _, first, inv = np.unique(np.round(V, 7), axis=0, return_index=True, return_inverse=True)
    Fw = inv[F]
    ok = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    return V[first], Fw[ok]


def decimate(V, F, target_faces):
    """Plain quadric edge collapse on a seamless mesh -- hits the target reliably."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(V, np.float64),
                               face_matrix=np.asarray(F, np.int32)), "src")
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces), qualitythr=0.3, preserveboundary=False,
        preservenormal=True, preservetopology=False, optimalplacement=True,
        planarquadric=True, autoclean=True)
    cm = ms.current_mesh()
    return np.asarray(cm.vertex_matrix()), np.asarray(cm.face_matrix())


def reatlas(V, F, tex_res, padding=4):
    """Fresh UV atlas for the decimated mesh. Returns (verts, faces, uv)."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(V, np.float32), np.asarray(F, np.uint32))
    po = xatlas.PackOptions()
    po.resolution = int(tex_res)
    po.padding = int(padding)
    po.bruteForce = False
    atlas.generate(pack_options=po)
    vmapping, indices, uvs = atlas[0]
    return V[vmapping.astype(np.int64)], indices.astype(np.int64), np.asarray(uvs, np.float64)


# ------------------------------------------------------------------------ bake

def bake_basecolor(Vn, Fn, UVn, srcV, srcF, srcUV, src_tex, res, knn=6):
    """Re-paint the new atlas from the original texture.

    Rasterise the new UV layout, take each texel's 3D point, find the closest point on
    the original surface (kd-tree over source triangle centroids + exact point-triangle
    tests on the candidates), read the source UV there, sample the source texture.
    """
    import torch
    import nvdiffrast.torch as dr
    from scipy.spatial import cKDTree
    from scipy.ndimage import distance_transform_edt
    from trimesh.triangles import closest_point as tri_closest, points_to_barycentric

    global _CTX
    if _CTX is None:
        _CTX = dr.RasterizeCudaContext()

    # UV -> clip space. nvdiffrast's row 0 is y=-1 and glTF puts v=0 at image row 0,
    # so y = 2v-1 lines the raster rows up with the output texture rows directly.
    # srcUV must already be in glTF convention (see gltf_uv).
    uv = np.asarray(UVn, np.float32)
    clip = np.stack([uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1,
                     np.zeros(len(uv), np.float32), np.ones(len(uv), np.float32)], 1)
    tri = torch.as_tensor(np.asarray(Fn, np.int32)).cuda()
    rast, _ = dr.rasterize(_CTX, torch.from_numpy(clip)[None].cuda(), tri, resolution=[res, res])
    pos, _ = dr.interpolate(torch.as_tensor(np.asarray(Vn, np.float32))[None].cuda(), rast, tri)
    covered = (rast[0, :, :, 3] > 0).cpu().numpy()
    pts = pos[0].cpu().numpy()[covered].astype(np.float64)

    tris = srcV[srcF]                                   # (M,3,3)
    _, cand = cKDTree(tris.mean(1)).query(pts, k=knn, workers=-1)
    best_d = np.full(len(pts), np.inf)
    best_t = np.zeros(len(pts), np.int64)
    for c in range(knn):
        idx = cand[:, c]
        q = tri_closest(tris[idx], pts)
        d = np.linalg.norm(q - pts, axis=1)
        take = d < best_d
        best_d[take], best_t[take] = d[take], idx[take]

    chosen = tris[best_t]
    q = tri_closest(chosen, pts)
    bary = points_to_barycentric(chosen, q, method="cross")
    # zero-area source triangles make the barycentric solve NaN; fall back to a corner
    bad = ~np.isfinite(bary).all(1)
    bary[bad] = (1.0, 0.0, 0.0)
    uv_src = np.einsum("ij,ijk->ik", bary, srcUV[srcF[best_t]])
    uv_src = np.nan_to_num(uv_src, nan=0.0, posinf=1.0, neginf=0.0)

    tex = np.asarray(src_tex.convert("RGB"), np.float32)
    H, W = tex.shape[:2]
    x = np.clip(uv_src[:, 0], 0, 1) * (W - 1)
    y = np.clip(uv_src[:, 1], 0, 1) * (H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    col = ((tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy) +
           (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)

    out = np.zeros((res, res, 3), np.float32)
    out[covered] = col
    # bleed the charts outward so bilinear filtering never samples empty atlas
    _, near = distance_transform_edt(~covered, return_indices=True)
    out = out[near[0], near[1]]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), float(covered.mean())


# ------------------------------------------------------------------------ texture

def as_webp(img, size, quality):
    img = img.convert("RGB")
    if img.width > size:
        img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=6)
    data = buf.getvalue()
    return data, len(data)


def as_webp_budget(img, size, quality, max_kb):
    """768px is worth it on most of these atlases, but a few (chrome, dense speckle)
    cost 1-2 MB at that size. Step down until the encode fits the byte budget."""
    ladder = [(size, quality), (size, quality - 10), (512, quality), (512, quality - 10),
              (384, quality - 10)]
    seen = []
    for sz, q in ladder:
        if (sz, q) in seen or q <= 0:
            continue
        seen.append((sz, q))
        data, n = as_webp(img, sz, q)
        if n <= max_kb * 1024:
            return data, n, f"{sz}px q{q}"
    return data, n, f"{sz}px q{q} over budget"


def rebuild_material(src_mat, baked, tex_size, tex_quality, mr_size, tex_max_kb):
    """Encode the re-baked baseColor; collapse a constant metallicRoughness into factors."""
    base, base_bytes, base_note = as_webp_budget(baked, tex_size, tex_quality, tex_max_kb)
    mat = {
        "baseColorFactor": [float(x) / 255.0 for x in np.asarray(src_mat.baseColorFactor)],
        "alphaMode": src_mat.alphaMode or "OPAQUE",
        "doubleSided": bool(src_mat.doubleSided),
        "metallicFactor": float(src_mat.metallicFactor if src_mat.metallicFactor is not None else 1.0),
        "roughnessFactor": float(src_mat.roughnessFactor if src_mat.roughnessFactor is not None else 1.0),
    }
    mr_img, mr_bytes, mr_note = None, 0, "none"
    mrt = src_mat.metallicRoughnessTexture
    if mrt is not None:
        arr = np.asarray(mrt.convert("RGB"), dtype=np.float32).reshape(-1, 3)
        spread = float(arr.std(0).max())
        if spread < 3.0:                     # flat map: exactly equivalent as factors
            mean = arr.mean(0) / 255.0
            mat["roughnessFactor"] *= float(mean[1])
            mat["metallicFactor"] *= float(mean[2])
            mr_note = f"folded to factors (std {spread:.2f})"
        else:
            mr_img, mr_bytes = as_webp(mrt, mr_size, 80)
            mr_note = f"{mr_size}px webp (std {spread:.2f})"
    return mat, base, base_bytes, base_note, mr_img, mr_bytes, mr_note


# --------------------------------------------------------------------- glb writer

def smooth_normals(V, F):
    """Area-weighted vertex normals, welded across UV seams so seams don't crease."""
    # vertices coincident in space but split in the atlas must share one normal
    _, weld = np.unique(np.round(V, 6), axis=0, return_inverse=True)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])   # |fn| ~ 2*area
    acc = np.zeros((weld.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, weld[F[:, k]], fn)
    n = acc[weld]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _pad4(b):
    return b + b"\0" * (-len(b) % 4)


def write_quantized_glb(path, V, F, UV, mat, base_img_bytes, mr_img_bytes):
    """Emit a GLB using KHR_mesh_quantization + EXT_texture_webp, by hand.

    trimesh's exporter writes float32 POSITION/TEXCOORD, uint32 indices and silently
    drops NORMAL, which costs ~2x the bytes for a worse-looking mesh. This is the
    gltfpack-shaped layout instead: int16 position (dequantised by the node scale),
    int8 normalized normals, uint16 normalized UVs, uint16 indices.
    """
    N = smooth_normals(V, F)
    lo, hi = V.min(0), V.max(0)
    centre = (lo + hi) / 2.0
    scale = max(float((hi - lo).max()), 1e-9) / 65534.0   # uniform: keeps normals exact
    qpos = np.clip(np.rint((V - centre) / scale), -32767, 32767).astype("<i2")
    qnrm = np.clip(np.rint(N * 127.0), -127, 127).astype("<i1")
    quv = np.rint(np.clip(UV, 0.0, 1.0) * 65535.0).astype("<u2")
    idx_u16 = len(V) <= 65535
    qidx = F.astype("<u2" if idx_u16 else "<u4")

    # pad the 3-component attributes out to 4-byte strides (WebGL/glTF alignment)
    pos_buf = np.zeros((len(V), 4), "<i2"); pos_buf[:, :3] = qpos
    nrm_buf = np.zeros((len(V), 4), "<i1"); nrm_buf[:, :3] = qnrm

    blobs, views, accessors = [], [], []
    cursor = 0

    def view(data, target=None, stride=None):
        nonlocal cursor
        data = _pad4(bytes(data))
        v = {"buffer": 0, "byteOffset": cursor, "byteLength": len(data)}
        if target: v["target"] = target
        if stride: v["byteStride"] = stride
        views.append(v); blobs.append(data); cursor += len(data)
        return len(views) - 1

    def acc(view_i, ctype, count, type_, **kw):
        a = {"bufferView": view_i, "componentType": ctype, "count": int(count), "type": type_}
        a.update(kw)
        accessors.append(a)
        return len(accessors) - 1

    a_idx = acc(view(qidx.tobytes(), 34963), 5123 if idx_u16 else 5125, F.size, "SCALAR")
    a_pos = acc(view(pos_buf.tobytes(), 34962, 8), 5122, len(V), "VEC3",
                min=qpos.min(0).tolist(), max=qpos.max(0).tolist())
    a_nrm = acc(view(nrm_buf.tobytes(), 34962, 4), 5120, len(V), "VEC3", normalized=True)
    a_uv = acc(view(quv.tobytes(), 34962, 4), 5123, len(V), "VEC2", normalized=True)

    images, textures = [], []

    def texture(img_bytes):
        images.append({"bufferView": view(img_bytes), "mimeType": "image/webp"})
        textures.append({"sampler": 0, "extensions": {"EXT_texture_webp": {"source": len(images) - 1}}})
        return {"index": len(textures) - 1}

    pbr = {
        "baseColorTexture": texture(base_img_bytes),
        "baseColorFactor": mat["baseColorFactor"],
        "metallicFactor": mat["metallicFactor"],
        "roughnessFactor": mat["roughnessFactor"],
    }
    if mr_img_bytes:
        pbr["metallicRoughnessTexture"] = texture(mr_img_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "compress_trimodal_glbs.py"},
        "extensionsUsed": ["KHR_mesh_quantization", "EXT_texture_webp"],
        "extensionsRequired": ["KHR_mesh_quantization", "EXT_texture_webp"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # the node undoes the position quantisation; uniform scale keeps normals valid
        "nodes": [{"mesh": 0, "translation": centre.tolist(), "scale": [scale] * 3}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv},
            "indices": a_idx, "material": 0, "mode": 4}]}],
        "materials": [{"pbrMetallicRoughness": pbr,
                       "alphaMode": mat["alphaMode"], "doubleSided": mat["doubleSided"]}],
        "textures": textures,
        "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": cursor}],
    }

    bin_chunk = b"".join(blobs)
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A)); fh.write(json_chunk)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942)); fh.write(bin_chunk)


# ------------------------------------------------------------------------- render

_CTX = None


def _clip(points, azimuth, elevation=20.0, dist=2.4, fov=35.0):
    a, e = np.deg2rad(azimuth), np.deg2rad(elevation)
    cam = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)]) * dist
    fwd = -cam / np.linalg.norm(cam)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = points - cam
    xyz = np.stack([rel @ right, rel @ up, rel @ fwd], axis=1)
    near, far = 0.05, 20.0
    focal = 1.0 / np.tan(np.deg2rad(fov) / 2.0)
    z = xyz[:, 2]
    return np.stack([focal * xyz[:, 0], focal * xyz[:, 1],
                     ((far + near) * z - 2 * near * far) / (far - near), z], 1).astype(np.float32)


def silhouettes(verts, faces, center, scale, azimuths, res=256):
    """Binary coverage masks under a *shared* normalisation so two meshes line up."""
    global _CTX
    import torch
    import nvdiffrast.torch as dr

    if _CTX is None:
        _CTX = dr.RasterizeCudaContext()
    pts = (np.asarray(verts, np.float32) - center) / scale
    tri = torch.as_tensor(np.asarray(faces, np.int32)).cuda()
    out = []
    for az in azimuths:
        pos = torch.from_numpy(_clip(pts, az))[None].cuda()
        rast, _ = dr.rasterize(_CTX, pos, tri, resolution=[res, res])
        out.append((rast[0, :, :, 3] > 0).cpu().numpy())
    return np.stack(out)


def silhouette_iou(a, b):
    inter = np.logical_and(a, b).sum((1, 2)).astype(np.float64)
    union = np.logical_or(a, b).sum((1, 2)).astype(np.float64)
    return float(np.mean(inter / np.maximum(union, 1)))


# ---------------------------------------------------------------------------- run

def compress_one(path, out_path, faces, tex, tex_q, mr_size, tex_max_kb, check=True):
    src = load_glb(path)
    srcV = np.asarray(src.vertices, np.float64)
    srcF = np.asarray(src.faces, np.int64)
    srcUV = gltf_uv(src)

    Vw, Fw = weld(srcV, srcF)
    Vd, Fd = decimate(Vw, Fw, faces)
    V, F, UV = reatlas(Vd, Fd, tex)
    baked, covered = bake_basecolor(V, F, UV, srcV, srcF, srcUV,
                                    src.visual.material.baseColorTexture, tex)

    mat, base_img, base_bytes, base_note, mr_img, mr_bytes, mr_note = rebuild_material(
        src.visual.material, baked, tex, tex_q, mr_size, tex_max_kb)
    write_quantized_glb(out_path, V, F, UV, mat, base_img, mr_img)

    rec = {
        "file": os.path.basename(out_path),
        "src_bytes": os.path.getsize(path),
        "out_bytes": os.path.getsize(out_path),
        "src_faces": int(len(src.faces)), "out_faces": int(len(F)),
        "src_verts": int(len(src.vertices)), "out_verts": int(len(V)),
        "atlas_fill": round(covered, 3),
        "basecolor_bytes": base_bytes, "basecolor": base_note,
        "mr_bytes": mr_bytes, "mr": mr_note,
    }
    if check:
        # read the file back so the check covers the quantisation + node transform too,
        # not just the decimation
        rt = trimesh.load(out_path, process=False, force="mesh")
        lo, hi = src.vertices.min(0), src.vertices.max(0)
        center, scale = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)
        az = np.arange(0, 360, 45)
        rec["silhouette_iou"] = round(silhouette_iou(
            silhouettes(src.vertices, src.faces, center, scale, az),
            silhouettes(rt.vertices, rt.faces, center, scale, az)), 4)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", type=int, default=25000)
    ap.add_argument("--tex", type=int, default=768)
    ap.add_argument("--tex-quality", type=int, default=82)
    ap.add_argument("--tex-max-kb", type=int, default=140)
    ap.add_argument("--mr", type=int, default=128)
    ap.add_argument("--glob", default="*.glb")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(SRC_DIR, args.glob))):
        name = os.path.basename(path)
        tgt = FACE_OVERRIDES.get(name[:-4], args.faces)
        rec = compress_one(path, os.path.join(OUT_DIR, name), tgt, args.tex,
                           args.tex_quality, args.mr, args.tex_max_kb, check=not args.no_check)
        rec["target_faces"] = tgt
        rows.append(rec)
        print(f"{name:<22} {rec['src_bytes']/1e6:5.2f} MB -> {rec['out_bytes']/1e6:5.3f} MB "
              f"| {rec['src_faces']:>7,}f -> {rec['out_faces']:>6,}f "
              f"| {rec['out_verts']:>6,}v | sil-IoU {rec.get('silhouette_iou', float('nan')):.4f} "
              f"| atlas {rec['atlas_fill']:.2f} "
              f"| base {rec['basecolor']} {rec['basecolor_bytes']//1024}K | mr: {rec['mr']}")

    tot_src = sum(r["src_bytes"] for r in rows)
    tot_out = sum(r["out_bytes"] for r in rows)
    print(f"\n{len(rows)} files: {tot_src/1e6:.1f} MB -> {tot_out/1e6:.1f} MB "
          f"({tot_src/max(tot_out,1):.0f}x smaller)")
    worst = sorted((r for r in rows if "silhouette_iou" in r), key=lambda r: r["silhouette_iou"])[:5]
    for r in worst:
        print(f"  worst silhouette: {r['file']} {r['silhouette_iou']:.4f}")
    json.dump({"params": vars(args), "files": rows,
               "total_src_bytes": tot_src, "total_out_bytes": tot_out},
              open(REPORT, "w"), indent=1)
    print("report ->", REPORT)


if __name__ == "__main__":
    main()
