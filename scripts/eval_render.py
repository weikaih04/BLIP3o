"""Final-effect eval for the shape-512 conditioning ablation: each arm's trained flow
samples the shape SLAT on FIXED GT coords (isolates the conditioning mechanism — the SS
structure is held constant), decodes to a mesh, renders views, and computes occupancy-IoU
vs the GT mesh. Produces a per-asset comparison grid [GT | ① | ② | ③].

Sampling replicates TRELLIS FlowEulerSampler exactly: t_seq linspace(1,0,steps+1) with
rescale, x_prev = x_t - (t-t_prev)·v, CFG v = s·v_pos + (1-s)·v_neg (neg = zeroed taps =
our training cfg-drop null). Coords stay = GT throughout (we only ablate the SLAT feats).
"""
import os, sys, json, argparse
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/sfr/weikaih/3dgen/model/BLIP3o")
import numpy as np
import torch
from PIL import Image

from trellis2_blip3o import _paths  # noqa
from trellis2.modules import sparse as sp  # type: ignore
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         load_norm_stats, SHAPE_SLAT_CONFIG_PATH)
from trellis2.utils import render_utils  # type: ignore

TAP_DIR = "/opt/dlami/nvme/vp1taps_v22"
ARMS = {"cross": "runs/ablate_cross/ckpt_final.pt",
        "mmdit": "runs/ablate_mmdit/ckpt_6000.pt",
        "plkv":  "runs/ablate_plkv/ckpt_final.pt"}
CKPT_ROOT = "/fsx/sfr/weikaih/3dgen/model/BLIP3o"


def build_arm(arch, ckpt):
    from scripts.ablate_shape import ArmModel
    m = ArmModel(arch).cuda().eval()
    sd = torch.load(os.path.join(CKPT_ROOT, ckpt), map_location="cuda")["model"]
    m.load_state_dict(sd, strict=True)
    return m


@torch.no_grad()
def sample_slat(arm, coords, taps, steps=25, cfg=3.0):
    """Euler-sample SLAT feats on fixed coords. taps (1,K,L,2048)."""
    dev = "cuda"
    N = coords.shape[0]
    x = sp.SparseTensor(torch.randn(N, 32, device=dev), coords.to(dev))  # fp32 (input_layer is fp32)
    mask = torch.ones(1, taps.shape[2], dtype=torch.bool, device=dev)
    taps_z = torch.zeros_like(taps)
    t_seq = np.linspace(1, 0, steps + 1).tolist()
    for i in range(steps):
        t, t_prev = t_seq[i], t_seq[i + 1]
        t_in = torch.tensor([1000 * t], device=dev, dtype=torch.float32)
        v_pos = arm(x, t_in, taps, mask).feats.float()
        if cfg != 1.0:
            v_neg = arm(x, t_in, taps_z, mask).feats.float()
            v = cfg * v_pos + (1 - cfg) * v_neg
        else:
            v = v_pos
        x = x.replace(x.feats.float() - (t - t_prev) * v)
    return x


@torch.no_grad()
def decode_render(decoder, slat_feats, coords, shape_norm, res=512, nviews=2, render_res=None):
    """Denormalize SLAT → decode mesh → render nviews. Returns (list[PIL], occ_coords)."""
    std = shape_norm["std"].to("cuda"); mean = shape_norm["mean"].to("cuda")
    feats = slat_feats.float() * std + mean
    slat = sp.SparseTensor(feats, coords.cuda())  # decoder handles its own cast
    decoder.set_resolution(res)
    out = decoder(slat, return_subs=False)
    mesh = out[0] if isinstance(out, (list, tuple)) else out
    try:
        mesh.fill_holes()
    except Exception:
        pass
    rd = render_utils.render_snapshot(mesh, resolution=(render_res or 384), nviews=nviews)
    key = "normal" if "normal" in rd else ("shaded" if "shaded" in rd else list(rd)[0])
    imgs = [Image.fromarray(f) for f in rd[key]]
    occ = set(map(tuple, coords[:, 1:].cpu().numpy().tolist()))  # GT-coords occupancy proxy
    return imgs, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--arms", default="cross")     # comma list; "cross" for smoke
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--out", default="runs/cache_logs/eval_render")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arms = a.arms.split(",")

    shape_norm = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
    idx = json.load(open(os.path.join(TAP_DIR, "_shape_index.json")))
    decoder = build_sc_vae_shape_decoder_frozen().cuda().eval()

    # fixed held-out assets: first N tap files that have a shape latent
    files = []
    for d2 in sorted(os.listdir(TAP_DIR)):
        p2 = os.path.join(TAP_DIR, d2)
        if len(d2) == 2 and os.path.isdir(p2):
            for f in sorted(os.listdir(p2)):
                if f.endswith(".npz") and f[:-4] in idx:
                    files.append((f[:-4], os.path.join(p2, f)))
    files = files[:a.n]
    print(f"[eval] {len(files)} assets, arms={arms}, steps={a.steps} cfg={a.cfg}", flush=True)

    # GT reference: decode the RAW GT SLAT feats directly (pass through denorm-inverse so
    # decode_render's ×std+mean recovers the raw feats)
    if "gt" in arms:
        mean = shape_norm["mean"].cuda(); std = shape_norm["std"].cuda()
        for sha, fp in files:
            s = np.load(idx[sha])
            coords_xyz = torch.from_numpy(s["coords"]).int()
            coords = torch.cat([torch.zeros(coords_xyz.shape[0], 1, dtype=torch.int32), coords_xyz], 1)
            gt_raw = torch.from_numpy(s["feats"]).float().cuda()
            imgs, _ = decode_render(decoder, (gt_raw - mean) / std, coords, shape_norm)
            for v, im in enumerate(imgs):
                im.save(os.path.join(a.out, f"{sha[:8]}_gt_v{v}.png"))
            print(f"[eval] gt {sha[:8]} done", flush=True)

    for arch in [x for x in arms if x != "gt"]:
        m = build_arm(arch, ARMS[arch])
        for sha, fp in files:
            az = np.load(fp)
            taps = torch.from_numpy(az["taps"]).view(torch.bfloat16).float().unsqueeze(0).cuda()
            s = np.load(idx[sha])
            coords_xyz = torch.from_numpy(s["coords"]).int()
            coords = torch.cat([torch.zeros(coords_xyz.shape[0], 1, dtype=torch.int32), coords_xyz], 1)
            slat = sample_slat(m, coords, taps, steps=a.steps, cfg=a.cfg)
            imgs, _ = decode_render(decoder, slat.feats, coords, shape_norm)
            for v, im in enumerate(imgs):
                im.save(os.path.join(a.out, f"{sha[:8]}_{arch}_v{v}.png"))
            print(f"[eval] {arch} {sha[:8]} done", flush=True)
        del m; torch.cuda.empty_cache()

    # assemble grid: rows = assets, cols = [gt cross mmdit plkv] (view 0)
    col = [c for c in ["gt", "cross", "mmdit", "plkv"] if c in arms]
    cell = 256
    grid = Image.new("RGB", (cell * len(col), cell * len(files)), (240, 240, 240))
    for r, (sha, _) in enumerate(files):
        for c, arch in enumerate(col):
            p = os.path.join(a.out, f"{sha[:8]}_{arch}_v0.png")
            if os.path.exists(p):
                grid.paste(Image.open(p).resize((cell, cell)), (c * cell, r * cell))
    grid.save(os.path.join(a.out, "_GRID.png"))
    print(f"[eval] GRID saved cols={col}", flush=True)
    print("[eval] DONE", flush=True)


if __name__ == "__main__":
    main()
