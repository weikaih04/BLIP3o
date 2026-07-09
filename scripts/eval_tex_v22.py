"""Texture-flow showcase: sample tex SLAT on GT shape (fixed-structure), decode to PBR,
render shaded under an envmap from the input-view camera.

Grid: [input | GT textured (GT shape+tex decoded) | old-view tex model | good-view tex model].
The tex flow is SHAPE-CONDITIONED: x_in = concat[x_t(32) ; shape_z(32, tex-config norm)].
"""
import os, sys, json, argparse
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/sfr/weikaih/3dgen/model/BLIP3o")
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

from trellis2_blip3o import _paths  # noqa
from trellis2.modules import sparse as sp  # type: ignore
from trellis2.representations import MeshWithVoxel  # type: ignore
from trellis2.renderers import EnvMap  # type: ignore
from trellis2.utils import render_utils as t2render  # type: ignore
import utils3d  # type: ignore
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen,
                                         load_norm_stats, SHAPE_SLAT_CONFIG_PATH,
                                         TEX_SLAT_CONFIG_PATH, DEFAULT_TEX_SLAT)
from trellis2_blip3o.connector import TRELLIS2Connector
from scripts.eval_fusion_v22 import (build_cond, cam_from_transforms, input_image,
                                     good_view_b)
import scripts.eval_fusion_v22 as EV

PBR = {'base_color': slice(0, 3), 'metallic': slice(3, 4), 'roughness': slice(4, 5), 'alpha': slice(5, 6)}
HDR = "/fsx/sfr/weikaih/3dgen/model/third_party_3d_gen/TRELLIS.2/assets/hdri/forest.exr"
COND_ROOT = os.environ.get("EVAL_COND_ROOT", "/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1")
MANI = os.environ.get("EVAL_MANI", "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_all.jsonl")


def load_tex_flow(ckpt_dir, use_ema=True):
    from trellis2 import models as t2models  # type: ignore
    flow = t2models.from_pretrained(DEFAULT_TEX_SLAT)
    sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    if use_ema:
        sd.update(load_file(os.path.join(ckpt_dir, "ema.safetensors")))
    fsd = {k[len("tex_slat_512."):]: v for k, v in sd.items() if k.startswith("tex_slat_512.")}
    missing, _ = flow.load_state_dict(fsd, strict=False)
    conn = TRELLIS2Connector(2048, flow.cond_channels)
    conn.load_state_dict({k[len("diffusion_connector."):]: v for k, v in sd.items()
                          if k.startswith("diffusion_connector.")}, strict=True)
    dve = sd.get("dino_view_embed")
    print(f"[load-tex] {os.path.basename(ckpt_dir)}: flow {len(fsd)} (missing={len(missing)})", flush=True)
    return flow.cuda().eval(), conn.cuda().eval().float(), (dve.cuda() if dve is not None else None)


@torch.no_grad()
def sample_tex(flow, cond, uncond, coords, shape_z_feats, steps=25, cfg=3.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    N = coords.shape[0]
    x = torch.randn(N, 32, generator=g, device="cuda", dtype=torch.float32)
    sz = shape_z_feats.float()
    ts = np.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t, t_prev = float(ts[i]), float(ts[i + 1])
        tt = torch.tensor([t * 1000.0], device="cuda")
        xin = sp.SparseTensor(torch.cat([x, sz], 1), coords.cuda())
        vp = flow(xin, tt, cond).feats.float()
        vn = flow(xin, tt, uncond).feats.float()
        v = cfg * vp + (1 - cfg) * vn
        x = x - (t - t_prev) * v
    return x


def render_textured(shape_dec, tex_dec, coords, shape_raw, tex_raw, extr, intr, envmap,
                    render_res=1024):
    """shape_raw/tex_raw = DENORMALIZED latents. Returns shaded PIL from given camera."""
    slat = sp.SparseTensor(shape_raw.float(), coords.cuda())
    shape_dec.set_resolution(512)
    meshes, gsubs = shape_dec(slat, return_subs=True)
    mesh = meshes[0]
    try:
        mesh.fill_holes()
    except Exception:
        pass
    tex_vox = tex_dec(sp.SparseTensor(tex_raw.float(), coords.cuda()), guide_subs=gsubs) * 0.5 + 0.5
    mw = MeshWithVoxel(mesh.vertices, mesh.faces, origin=[-0.5, -0.5, -0.5],
                       voxel_size=1 / 512, coords=tex_vox.coords[:, 1:], attrs=tex_vox.feats,
                       voxel_shape=torch.Size([*tex_vox.shape, *tex_vox.spatial_shape]), layout=PBR)
    rd = t2render.render_frames(mw, [extr], [intr],
                                {"resolution": render_res, "bg_color": (1, 1, 1)}, envmap=envmap)
    key = "shaded" if "shaded" in rd else ("color" if "color" in rd else list(rd)[0])
    return Image.fromarray(rd[key][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fusion_tex_v22gv/checkpoint-16000")
    ap.add_argument("--ckpt_old", default="runs/fusion_tex_v22f/checkpoint-16000")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--out", default="runs/cache_logs/eval_tex")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = []
    with open(MANI) as f:
        for line in f:
            r = json.loads(line)
            v = r.get("vlm") or {}
            any_ok = os.environ.get("EVAL_ANY", "0") == "1"
            if any_ok or (v.get("part_complexity", 0) >= 7 and v.get("structural_score", 0) >= 7
                          and v.get("texture_score", 0) >= 6):
                ed = os.path.join(COND_ROOT, r["sha256"][:2], r["sha256"])
                if os.path.exists(os.path.join(ed, "v000.npz")) and r.get("pbr_latent_512"):
                    recs.append(r)
            if len(recs) >= a.n:
                break
    print(f"[tex-eval] {len(recs)} assets", flush=True)

    flow, conn, dve = load_tex_flow(a.ckpt)
    flow_o, conn_o, dve_o = load_tex_flow(a.ckpt_old)
    shape_dec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tex_dec = build_sc_vae_tex_decoder_frozen().cuda().eval()
    hdr = cv2.cvtColor(cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(hdr).cuda())

    tex_norm = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    tex_shape_norm = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    tm, tsd = tex_norm["mean"].cuda(), tex_norm["std"].cuda()
    sm, ssd = tex_shape_norm["mean"].cuda(), tex_shape_norm["std"].cuda()

    cell, hdrh = 640, 72
    _font = ImageFont.truetype("/fsx/sfr/weikaih/miniconda3/envs/blip3o_trellis/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 44)
    cols = (["input", "GT textured", "qwen-only", "qwen+dino"]
            if os.environ.get("OLD_QWEN_ONLY") == "1"
            else ["input", "GT textured", "old-view tex", "good-view tex"])
    grid = Image.new("RGB", (cell * 4, hdrh + cell * len(recs)), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    for c, label in enumerate(cols):
        d.text((c * cell + 12, 14), label, fill=(10, 10, 10), font=_font)

    for ri, r in enumerate(recs):
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        gt_s = np.load(r["shape_latent_512"])
        gt_t = np.load(r["pbr_latent_512"])
        cx = torch.from_numpy(gt_s["coords"]).int()
        coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
        shape_raw = torch.from_numpy(gt_s["feats"]).float().cuda()
        tex_raw_gt = torch.from_numpy(gt_t["feats"]).float().cuda()
        extr, intr = cam_from_transforms(r["renders_dir"])
        y = hdrh + ri * cell
        grid.paste(input_image(r["renders_dir"]).resize((cell, cell), Image.LANCZOS), (0, y))
        img = render_textured(shape_dec, tex_dec, coords, shape_raw, tex_raw_gt, extr, intr, envmap)
        grid.paste(img.resize((cell, cell), Image.LANCZOS), (cell, y))
        cond, uncond = build_cond(conn, dve, os.path.join(COND_ROOT, sha[:2], sha))
        _qo = os.environ.get("OLD_QWEN_ONLY", "0") == "1"
        cond_o, uncond_o = build_cond(conn_o, dve_o, os.path.join(COND_ROOT, sha[:2], sha), qwen_only=_qo)
        shape_z = (shape_raw - sm) / ssd
        for col, (fl, cc, uu) in enumerate([(flow_o, cond_o, uncond_o), (flow, cond, uncond)]):
            tex_n = sample_tex(fl, cc, uu, coords, shape_z, steps=a.steps, cfg=a.cfg)
            tex_raw = tex_n.cuda() * tsd + tm
            img = render_textured(shape_dec, tex_dec, coords, shape_raw, tex_raw, extr, intr, envmap)
            grid.paste(img.resize((cell, cell), Image.LANCZOS), ((2 + col) * cell, y))
        print(f"[tex-eval] {ri+1}/{len(recs)} {sha[:12]} done", flush=True)

    out = os.path.join(a.out, "_TEX_V22.png")
    grid.save(out)
    print(f"[tex-eval] saved {out}", flush=True)


if __name__ == "__main__":
    main()
