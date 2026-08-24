"""IM learnability probe eval — does conditioning on 4 views beat 1 view for SS geometry?

For each of the 14 held-out assets, compute SS occupancy IoU (vs GT) under three conds:
  A) I1 baseline   : deployed fusion SS ckpt-22000 + single-view fusion cond (v22_heldout/v000)
  B) IM probe I1   : the IM-trained probe ckpt + single-view cond (within-model 1-view control)
  C) IM probe IM   : the IM-trained probe ckpt + 4-view joint cond (v22_heldout_im4/m00)
KEY METRIC: mean IoU(C) > mean IoU(A)  → more views buy more geometric fidelity (learnable).

Also renders a grid for the first N_GRID assets:
  [ 4 input views | IM-probe 3-view occ proj (C) | I1-baseline occ proj (A) | GT occ proj ]

Env:
  SS_CKPT_IM  = probe ckpt dir (e.g. runs/im_ss_probe/checkpoint-6000)   [required]
  SS_CKPT_I1  = deployed I1 baseline ckpt (default runs/fusion_ss_dpos_2n/checkpoint-22000)
  COND_I1     = I1 cond cache root (default .../v22_heldout)
  COND_IM     = IM cond cache root (default /fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_im4)
  MANI        = held-out manifest (default /fsx/home/weikai.huang/3dgen/im_probe/heldout14.jsonl)
  OUT_TAG     = suffix for the output png/json (default = basename of SS_CKPT_IM)
  N_GRID      = assets to render in the grid (default 6)
"""
import os, sys, json
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from trellis2_blip3o import _paths  # noqa
from trellis2 import models as t2models  # type: ignore
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # type: ignore
import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import build_cond, good_view_b
from scripts.export_glb_fullchain import load_ss_flow, SSDEC

ROOT = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
SS_CKPT_IM = os.environ["SS_CKPT_IM"]
SS_CKPT_I1 = os.environ.get("SS_CKPT_I1", f"{ROOT}/runs/fusion_ss_dpos_2n/checkpoint-22000")
COND_I1 = os.environ.get("COND_I1", "/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_heldout")
COND_IM = os.environ.get("COND_IM", "/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_im4l")
MANI = os.environ.get("MANI", "/fsx/home/weikai.huang/3dgen/im_probe/heldout14.jsonl")
SEED = int(os.environ.get("SEED", "0"))
SHARD = int(os.environ.get("SHARD", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
_tag = os.path.basename(SS_CKPT_IM.rstrip("/"))
if SEED:
    _tag += f"_s{SEED}"
if NUM_SHARDS > 1:
    _tag += f"_sh{SHARD}of{NUM_SHARDS}"
OUT_TAG = os.environ.get("OUT_TAG", _tag)
N_GRID = int(os.environ.get("N_GRID", "6"))
# Official SS sampler settings (= fused_fullchain_eval / render_ss_occ / eval_ss_progress)
SS_OFF = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
              guidance_interval=[0.6, 1.0], rescale_t=5.0)
FONT_PATH = ("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
             "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")


def build_cond_im(conn, dve, entry_dir):
    """4-view joint fusion cond from an IM combo m00 entry (dino_view_ids → per-view dve)."""
    a = np.load(os.path.join(entry_dir, "m00.npz"))
    qwen = torch.from_numpy(a["hidden"]).float().cuda()          # (Tq, 2048) joint 4-view
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()     # (Td=4*1029, 1024)
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()    # (Td,) per-token view ordinal
    with torch.no_grad():
        cq = conn(qwen[None])
        c0 = conn(torch.zeros_like(qwen)[None])
        if getattr(conn, "pos_stamp", None) is not None:
            from trellis2_blip3o.pos_stamp import IMG_SPAN_FULL
            cq = conn.pos_stamp(cq, IMG_SPAN_FULL)
            c0 = conn.pos_stamp(c0, IMG_SPAN_FULL)
        # QWEN-segment view identity — must mirror training exactly (flow_heads applies the
        # SAME dve to the qwen tokens). Without it the model is evaluated missing a signal it
        # was trained with. Layout is the cache's fixed one: 292 tokens, four 64-token image
        # blocks at 10/76/142/208 (stride 66); text/structural tokens get no code.
        if dve is not None and qwen.shape[0] >= 272:
            qv = torch.full((qwen.shape[0],), -1, dtype=torch.long, device=cq.device)
            for v in range(4):
                qv[10 + 66 * v: 10 + 66 * v + 64] = v
            add = dve[qv.clamp_min(0)].float() * (qv >= 0).unsqueeze(-1).float()
            cq = cq + add[None]
    # trellis2_blip3o.eval_cond wraps build_unified_cond — the training loop's own
    # builder — so eval and training cannot disagree about the dino segment, the
    # view embedding, or which drop is CFG.
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    cond, uncond = cond_uncond_from_tensors(conn, qwen, qmask, dino, dmask, dve,
                                            dino_view_ids=vids, qwen_view_ids=qv)
    return cond, uncond, a["views"]


@torch.no_grad()
def sample_occ(flow, sampler, ssdec, cond, uncond, seed=SEED):
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution,
                        generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = sampler.sample(flow, noise, cond=cond, neg_cond=uncond, verbose=False, **SS_OFF).samples
    return (ssdec(z) > 0)[0, 0].cpu().numpy()


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else -1.0


def proj(occ):
    """3 orthographic max-projections stacked → one 384x128 RGB."""
    views = [occ.max(ax) for ax in (2, 1, 0)]
    canvas = np.zeros((64, 64 * 3), np.uint8)
    for i, v in enumerate(views):
        canvas[:, i * 64:(i + 1) * 64] = (v * 255).astype(np.uint8)
    return Image.fromarray(canvas).resize((64 * 3 * 2, 64 * 2), Image.NEAREST).convert("RGB")


def load_view_imgs(renders_dir, view_ids):
    out = []
    for v in view_ids:
        p = None
        for ext in ("webp", "png", "jpg"):
            cand = os.path.join(renders_dir, f"{int(v):03d}.{ext}")
            if os.path.isfile(cand):
                p = cand; break
        if p:
            out.append(Image.open(p).convert("RGB"))
    return out


def main():
    recs = [json.loads(l) for l in open(MANI)]
    if NUM_SHARDS > 1:
        recs = recs[SHARD::NUM_SHARDS]
    print(f"[eval] {len(recs)} held-out assets (shard {SHARD}/{NUM_SHARDS}); "
          f"probe={SS_CKPT_IM} seed={SEED}", flush=True)

    ssdec = t2models.from_pretrained(SSDEC).cuda().eval()
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    im_flow, im_conn, im_dve = load_ss_flow(SS_CKPT_IM)     # IM-trained probe
    i1_flow, i1_conn, i1_dve = load_ss_flow(SS_CKPT_I1)     # deployed I1 baseline

    rows = []
    grid_data = []
    for r in recs:
        sha = r["sha256"]; s8 = sha[:8]
        EV._VIEW_FILE = good_view_b(sha)
        ed_i1 = os.path.join(COND_I1, sha[:2], sha)
        ed_im = os.path.join(COND_IM, sha[:2], sha)
        if not os.path.exists(os.path.join(ed_im, "m00.npz")):
            print(f"[skip] {s8}: no IM cond entry", flush=True); continue
        gz = torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None]
        with torch.no_grad():
            gt = (ssdec(gz) > 0)[0, 0].cpu().numpy()

        # A) I1 baseline (deployed ckpt-22000, single view)
        cA, uA = build_cond(i1_conn, i1_dve, ed_i1, qwen_only=False)
        occA = sample_occ(i1_flow, sampler, ssdec, cA, uA)
        # B) IM probe with single-view cond (within-model 1-view control)
        cB, uB = build_cond(im_conn, im_dve, ed_i1, qwen_only=False)
        occB = sample_occ(im_flow, sampler, ssdec, cB, uB)
        # C) IM probe with 4-view cond
        cC, uC, views = build_cond_im(im_conn, im_dve, ed_im)
        occC = sample_occ(im_flow, sampler, ssdec, cC, uC)

        ioA, ioB, ioC = iou(occA, gt), iou(occB, gt), iou(occC, gt)
        rows.append((s8, ioA, ioB, ioC))
        print(f"[iou] {s8}  I1base={ioA:.3f}  probe_I1={ioB:.3f}  probe_IM(4v)={ioC:.3f}  "
              f"views={list(int(v) for v in views)}", flush=True)
        grid_data.append((s8, r["renders_dir"], list(int(v) for v in views), occC, occA, gt, ioC, ioA))

    # ---- IoU summary table ----
    A = np.array([x[1] for x in rows]); B = np.array([x[2] for x in rows]); C = np.array([x[3] for x in rows])
    m = (A >= 0) & (B >= 0) & (C >= 0)
    summ = {
        "probe_ckpt": SS_CKPT_IM, "n": int(m.sum()), "seed": SEED, "mani": MANI,
        "cond_i1": COND_I1, "cond_im": COND_IM,
        "shard": SHARD, "num_shards": NUM_SHARDS,
        "mean_I1_baseline": float(A[m].mean()), "mean_probe_I1": float(B[m].mean()),
        "mean_probe_IM_4view": float(C[m].mean()),
        "delta_IM_minus_I1base": float(C[m].mean() - A[m].mean()),
        "delta_IM_minus_probeI1": float(C[m].mean() - B[m].mean()),
        "win_IM_gt_I1base_per_asset": int(((C > A) & m).sum()),
        "per_asset": [{"sha8": s, "I1_baseline": a, "probe_I1": b, "probe_IM_4view": c}
                      for (s, a, b, c) in rows],
    }
    out_json = f"{ROOT}/runs/cache_logs/im_probe_iou_{OUT_TAG}.json"
    json.dump(summ, open(out_json, "w"), indent=2)
    print("\n=== IoU SUMMARY ===")
    print(f"  n={summ['n']}  I1_baseline={summ['mean_I1_baseline']:.3f}  "
          f"probe_I1={summ['mean_probe_I1']:.3f}  probe_IM(4v)={summ['mean_probe_IM_4view']:.3f}")
    print(f"  KEY: IM(4v) - I1_baseline = {summ['delta_IM_minus_I1base']:+.3f} ; "
          f"IM(4v) - probe_I1 = {summ['delta_IM_minus_probeI1']:+.3f} ; "
          f"IM>I1base on {summ['win_IM_gt_I1base_per_asset']}/{summ['n']} assets")
    print(f"  wrote {out_json}", flush=True)

    # ---- grid ----
    grid_data = grid_data[:N_GRID]
    if grid_data:
        cell = 256
        tfont = ImageFont.truetype(FONT_PATH, 22); rfont = ImageFont.truetype(FONT_PATH, 18)
        occ_w = 64 * 3 * 2  # 384
        cols_w = [cell, occ_w, occ_w, occ_w]
        hdr = 40
        W = sum(cols_w) + 30
        H = hdr + cell * len(grid_data)
        grid = Image.new("RGB", (W, H), (250, 250, 250))
        d = ImageDraw.Draw(grid)
        xs = [0]
        for w in cols_w[:-1]:
            xs.append(xs[-1] + w + 10)
        labels = ["4 input views", "IM probe (4v)", "I1 baseline (1v)", "GT"]
        for x, lab in zip(xs, labels):
            d.text((x + 6, 10), lab, fill=(10, 10, 10), font=tfont)
        for ri, (s8, rd, views, occC, occA, gt, ioC, ioA) in enumerate(grid_data):
            y = hdr + ri * cell
            # col0: 2x2 tile of the 4 input views
            imgs = load_view_imgs(rd, views)
            tile = Image.new("RGB", (cell, cell), (255, 255, 255))
            for k, im in enumerate(imgs[:4]):
                im = im.resize((cell // 2, cell // 2), Image.LANCZOS)
                tile.paste(im, ((k % 2) * (cell // 2), (k // 2) * (cell // 2)))
            grid.paste(tile, (xs[0], y))
            grid.paste(proj(occC), (xs[1], y + (cell - 128) // 2))
            grid.paste(proj(occA), (xs[2], y + (cell - 128) // 2))
            grid.paste(proj(gt),   (xs[3], y + (cell - 128) // 2))
            d.text((xs[0] + 4, y + cell - 24), f"{s8}", fill=(0, 0, 0), font=rfont)
            d.text((xs[1] + 4, y + cell - 24), f"IoU {ioC:.3f}", fill=(0, 90, 0), font=rfont)
            d.text((xs[2] + 4, y + cell - 24), f"IoU {ioA:.3f}", fill=(120, 0, 0), font=rfont)
        out_png = f"{ROOT}/runs/cache_logs/im_probe_grid_{OUT_TAG}.png"
        grid.save(out_png)
        print(f"[grid] saved {out_png}", flush=True)
    print("IM_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
