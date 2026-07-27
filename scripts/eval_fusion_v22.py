"""Render-eval a v22 fusion checkpoint (shape flow) — hub-report style grid:
[input image | GT normal render | fusion@ckpt normal render] x N assets.

Fixed-GT-structure protocol (same as the conditioning-ablation report): coords come
from the GT shape latent; only SLAT feats are sampled. Cond = fusion [DINOv3 ; connector(v2.2)]
from the offline v22 cache (v000 merged entries), CFG uncond = [zeros ; connector(0)].
EMA weights overlaid (the released-TRELLIS convention).
"""
import os, sys, json, argparse
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/sfr/weikaih/3dgen/model/BLIP3o")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
_VIEW_FILE = "008.webp"
from PIL import Image, ImageDraw, ImageFont
from trellis2_blip3o import _paths  # noqa
from trellis2.modules import sparse as sp  # type: ignore
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         load_norm_stats, SHAPE_SLAT_CONFIG_PATH)
from benchmarks.checkpoint import load_connector, load_state_dict
from scripts.eval_render import decode_render
from trellis2.utils import render_utils as t2render  # type: ignore
import utils3d  # type: ignore

COND_ROOT = os.environ.get("EVAL_COND_ROOT", "/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1")
MANI = os.environ.get("EVAL_MANI", "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_all.jsonl")
# base pretrained shape-SLAT flow. Resolve via tr2_modules (_paths.CHECKPOINTS_ROOT) so the
# path follows the repo location — the old hardcoded /fsx/sfr/weikaih/... is dead on the
# xgen-mm cluster (/fsx/home/weikai.huang/...).
from trellis2_blip3o.tr2_modules import DEFAULT_SHAPE_SLAT as _DEFAULT_SHAPE_SLAT
SHAPE_CKPT = os.environ.get("EVAL_BASE_SHAPE_SLAT", _DEFAULT_SHAPE_SLAT)


def load_flow_and_connector(ckpt_dir, use_ema=True):
    from trellis2 import models as t2models  # type: ignore
    flow = t2models.from_pretrained(SHAPE_CKPT)
    sd = load_state_dict(ckpt_dir, use_ema=use_ema)
    flow_sd = {k[len("shape_slat_512."):]: v for k, v in sd.items()
               if k.startswith("shape_slat_512.")}
    missing, unexpected = flow.load_state_dict(flow_sd, strict=False)
    conn, dve, ckcfg = load_connector(ckpt_dir, state=sd,
                                      vlm_hidden_dim=2048,
                                      cond_dim=flow.cond_channels)
    conn_sd = {k for k in sd if k.startswith("diffusion_connector.")}
    print(f"[load] flow {len(flow_sd)} tensors (missing={len(missing)}), "
          f"conn={ckcfg['cond_adapter']} ({len(conn_sd)} tensors), "
          f"pos_stamp={ckcfg.get('cond_pos_stamp', False)}, ema={use_ema}", flush=True)
    return flow.cuda().eval(), conn.cuda().eval().float(), (dve.cuda() if dve is not None else None)


def build_cond(conn, dve, entry_dir, qwen_only=False):
    a = np.load(os.path.join(entry_dir, "v000.npz"))
    qwen = torch.from_numpy(a["hidden"]).float().cuda()          # (Tq, 2048)
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()     # (Td, 1024)
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    with torch.no_grad():
        cq = conn(qwen[None])                                    # (1, Tq, 1024)
        c0 = conn(torch.zeros_like(qwen)[None])
        if getattr(conn, "pos_stamp", None) is not None:         # dpos stamp (pos_stamp.py) —
            from trellis2_blip3o.pos_stamp import IMG_SPAN_FULL  # full-seq span, BEFORE keep-indexing
            cq = conn.pos_stamp(cq, IMG_SPAN_FULL)
            c0 = conn.pos_stamp(c0, IMG_SPAN_FULL)               # uncond stamped too (train parity)
        dseg = dino[None]
        if dve is not None:
            dseg = dseg + dve[0][None, None].float()
        cond = torch.cat([dseg, cq], 1)                          # (1, Td+Tq, 1024)
        mask = torch.cat([dmask, qmask])[None]                   # (1, Td+Tq)
        # CFG uncond: zeros-DINO ; connector(0)  (flow_heads convention)
        uncond = torch.cat([torch.zeros_like(dseg), c0], 1)
    if qwen_only:
        # DINO segment ABSENT — the dino_drop training regime (keys masked off entirely)
        keep = qmask
        return cq[:, keep], c0[:, keep]
    # keep only masked-in tokens (bs=1 → simply index, no padding needed)
    keep = mask[0]
    return cond[:, keep], uncond[:, keep]


@torch.no_grad()
def sample_shape(flow, cond, uncond, coords, steps=25, cfg=3.0, sigma_min=1e-5, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    N = coords.shape[0]
    x = sp.SparseTensor(torch.randn(N, flow.in_channels, generator=g,
                                    device="cuda", dtype=torch.float32), coords.cuda())
    ts = np.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t, t_prev = float(ts[i]), float(ts[i + 1])
        tt = torch.tensor([t * 1000.0], device="cuda")
        vp = flow(x, tt, cond).feats.float()
        vn = flow(x, tt, uncond).feats.float()
        v = cfg * vp + (1 - cfg) * vn
        x = x.replace(x.feats - (t - t_prev) * v)
    return x




def cam_from_transforms(renders_dir, fname=None):
    fname = fname or _VIEW_FILE
    """input-view camera → TRELLIS (extrinsics, intrinsics). Blender/NeRF c2w (OpenGL,
    look -Z, up +Y) → OpenCV via negating Y/Z columns; w2c = inverse."""
    t = json.load(open(os.path.join(renders_dir, "transforms.json")))
    fr = next(f for f in t["frames"] if f["file_path"].endswith(fname))
    M = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
    M[:3, 1] *= -1; M[:3, 2] *= -1
    extr = torch.linalg.inv(M).cuda()
    fov = torch.tensor(float(fr["camera_angle_x"])).cuda()
    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    return extr, intr


def decode_render_cam(decoder, slat_feats, coords, shape_norm, extr, intr, res=512, render_res=1024):
    std = shape_norm["std"].cuda(); mean = shape_norm["mean"].cuda()
    slat = sp.SparseTensor(slat_feats.float() * std + mean, coords.cuda())
    decoder.set_resolution(res)
    out = decoder(slat, return_subs=False)
    mesh = out[0] if isinstance(out, (list, tuple)) else out
    try: mesh.fill_holes()
    except Exception: pass
    rd = t2render.render_frames(mesh, [extr], [intr], {"resolution": render_res, "bg_color": (0, 0, 0)})
    key = "normal" if "normal" in rd else list(rd)[0]
    return Image.fromarray(rd[key][0])

def good_view_b(sha):
    return f"{5 + (int(sha[:8], 16) + 3) % 7:03d}.webp"


def input_image(renders_dir):
    p = os.path.join(renders_dir, _VIEW_FILE)
    if not os.path.exists(p):
        c = sorted(f for f in os.listdir(renders_dir) if f.endswith(".webp"))
        p = os.path.join(renders_dir, c[0])
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fusion_shape_v22gv/checkpoint-16000")
    ap.add_argument("--ckpt_old", default="runs/fusion_shape_v22f/checkpoint-16000")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--out", default="runs/cache_logs/eval_fusion_v22")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # pick complex, high-quality assets (VLM axes in the manifest)
    recs = []
    with open(MANI) as f:
        for line in f:
            r = json.loads(line)
            v = r.get("vlm") or {}
            _thr = os.environ.get("EVAL_ANY", "0") == "1"
            if _thr or (v.get("part_complexity", 0) >= 7 and v.get("structural_score", 0) >= 7
                    and v.get("detail_complexity", 0) >= 6):
                ed = os.path.join(COND_ROOT, r["sha256"][:2], r["sha256"])
                if os.path.exists(os.path.join(ed, "v000.npz")):
                    recs.append(r)
            if len(recs) >= a.n:
                break
    print(f"[eval] {len(recs)} assets picked", flush=True)

    flow, conn, dve = load_flow_and_connector(a.ckpt)
    flow_o, conn_o, dve_o = load_flow_and_connector(a.ckpt_old)
    decoder = build_sc_vae_shape_decoder_frozen().cuda().eval()
    norm = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
    m_, sd_ = norm["mean"].cuda(), norm["std"].cuda()

    cell, hdr = 640, 72
    _font = ImageFont.truetype("/fsx/sfr/weikaih/miniconda3/envs/blip3o_trellis/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 44)
    cols = (["input (good view)", "GT", "qwen-only", "qwen+dino"]
            if os.environ.get("OLD_QWEN_ONLY") == "1"
            else ["input (good view)", "GT", "old-view model", "good-view model"])
    grid = Image.new("RGB", (cell * 4, hdr + cell * len(recs)), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    for c, label in enumerate(cols):
        d.text((c * cell + 12, 14), label, fill=(10, 10, 10), font=_font)

    for ri, r in enumerate(recs):
        sha = r["sha256"]
        global _VIEW_FILE
        _VIEW_FILE = good_view_b(sha)
        gt = np.load(r["shape_latent_512"])
        cx = torch.from_numpy(gt["coords"]).int()
        coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
        y = hdr + ri * cell
        grid.paste(input_image(r["renders_dir"]).resize((cell, cell), Image.LANCZOS), (0, y))
        extr, intr = cam_from_transforms(r["renders_dir"])
        gt_feats = torch.from_numpy(gt["feats"]).float().cuda()
        img = decode_render_cam(decoder, (gt_feats - m_) / sd_, coords, norm, extr, intr)
        grid.paste(img.resize((cell, cell), Image.LANCZOS), (cell, y))
        cond, uncond = build_cond(conn, dve, os.path.join(COND_ROOT, sha[:2], sha))
        _qo = os.environ.get("OLD_QWEN_ONLY", "0") == "1"
        cond_o, uncond_o = build_cond(conn_o, dve_o, os.path.join(COND_ROOT, sha[:2], sha), qwen_only=_qo)
        slat_o = sample_shape(flow_o, cond_o, uncond_o, coords, steps=a.steps, cfg=a.cfg)
        img = decode_render_cam(decoder, slat_o.feats, coords, norm, extr, intr)
        grid.paste(img.resize((cell, cell), Image.LANCZOS), (2 * cell, y))
        slat = sample_shape(flow, cond, uncond, coords, steps=a.steps, cfg=a.cfg)
        img = decode_render_cam(decoder, slat.feats, coords, norm, extr, intr)
        grid.paste(img.resize((cell, cell), Image.LANCZOS), (3 * cell, y))
        print(f"[eval] {ri+1}/{len(recs)} {sha[:12]} done", flush=True)

    out = os.path.join(a.out, "_FUSION_V22.png")
    grid.save(out)
    print(f"[eval] saved {out}", flush=True)


if __name__ == "__main__":
    main()
