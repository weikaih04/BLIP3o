"""Export interactive GLBs for the hub 3D viewer: per held-out asset, generate with the
good-view fusion models (shape+tex on GT coords) and export BOTH gen and GT as GLB.
Sizes kept web-friendly (decimation 200k, texture 2048, webp)."""
import os, sys, json, argparse
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("FUSED_MODULATE", "1")
os.environ.setdefault("EVAL_COND_ROOT", "/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/v22_heldout")
os.environ.setdefault("EVAL_MANI", "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/heldout_eval.jsonl")
sys.path.insert(0, "/fsx/sfr/weikaih/3dgen/model/BLIP3o")
import numpy as np
import torch

from trellis2_blip3o import _paths  # noqa
import o_voxel  # type: ignore
from trellis2.modules import sparse as sp  # type: ignore
from trellis2.representations import MeshWithVoxel  # type: ignore
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen,
                                         load_norm_stats, SHAPE_SLAT_CONFIG_PATH,
                                         TEX_SLAT_CONFIG_PATH)
import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import (load_flow_and_connector, build_cond, sample_shape,
                                     good_view_b, input_image)
from scripts.eval_tex_v22 import load_tex_flow, sample_tex

PBR = {'base_color': slice(0, 3), 'metallic': slice(3, 4), 'roughness': slice(4, 5), 'alpha': slice(5, 6)}
CR = os.environ["EVAL_COND_ROOT"]


def build_mw(sdec, tdec, coords, shape_raw, tex_raw):
    slat = sp.SparseTensor(shape_raw.float(), coords.cuda())
    sdec.set_resolution(512)
    meshes, gsubs = sdec(slat, return_subs=True)
    mesh = meshes[0]
    try:
        mesh.fill_holes()
    except Exception:
        pass
    tex_vox = tdec(sp.SparseTensor(tex_raw.float(), coords.cuda()), guide_subs=gsubs) * 0.5 + 0.5
    return MeshWithVoxel(mesh.vertices, mesh.faces, origin=[-0.5, -0.5, -0.5],
                         voxel_size=1 / 512, coords=tex_vox.coords[:, 1:], attrs=tex_vox.feats,
                         voxel_shape=torch.Size([*tex_vox.shape, *tex_vox.spatial_shape]), layout=PBR)


def export_glb(mw, path):
    glb = o_voxel.postprocess.to_glb(
        vertices=mw.vertices, faces=mw.faces, attr_volume=mw.attrs, coords=mw.coords,
        attr_layout=mw.layout, voxel_size=mw.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=200000, texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    glb.export(path, extension_webp=True)
    print(f"[glb] {path} ({os.path.getsize(path)/1e6:.1f}MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="runs/cache_logs/glb_viewer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for l in open(os.environ["EVAL_MANI"])][:a.n]

    flow, conn, dve = load_flow_and_connector("runs/fusion_shape_v22gv/checkpoint-16000")
    tflow, tconn, tdve = load_tex_flow("runs/fusion_tex_v22gv/checkpoint-16000")
    sdec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tdec = build_sc_vae_tex_decoder_frozen().cuda().eval()
    sn = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    tsn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    sm_, ssd_ = sn["mean"].cuda(), sn["std"].cuda()
    tm_, tsd_ = tn["mean"].cuda(), tn["std"].cuda()
    xm_, xsd_ = tsn["mean"].cuda(), tsn["std"].cuda()

    meta = []
    for r in recs:
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        gt = np.load(r["shape_latent_512"]); gtt = np.load(r["pbr_latent_512"])
        cx = torch.from_numpy(gt["coords"]).int()
        coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
        gs = torch.from_numpy(gt["feats"]).float().cuda()
        gtx = torch.from_numpy(gtt["feats"]).float().cuda()
        ed = os.path.join(CR, sha[:2], sha)
        # input image
        input_image(r["renders_dir"]).save(os.path.join(a.out, f"input_{sha[:12]}.webp"))
        # GT glb
        export_glb(build_mw(sdec, tdec, coords, gs, gtx), os.path.join(a.out, f"gt_{sha[:12]}.glb"))
        # generated: shape flow → generated shape SLAT; tex flow on GENERATED shape
        cond, uncond = build_cond(conn, dve, ed)
        slat = sample_shape(flow, cond, uncond, coords)
        gen_shape_raw = slat.feats.float() * ssd_ + sm_
        tcond, tuncond = build_cond(tconn, tdve, ed)
        tex_n = sample_tex(tflow, tcond, tuncond, coords, (gen_shape_raw - xm_) / xsd_)
        gen_tex_raw = tex_n.cuda() * tsd_ + tm_
        export_glb(build_mw(sdec, tdec, coords, gen_shape_raw, gen_tex_raw),
                   os.path.join(a.out, f"gen_{sha[:12]}.glb"))
        meta.append({"sha": sha[:12], "subset": r["subset"]})
        print(f"[glb] {sha[:12]} done", flush=True)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    print("[glb] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
