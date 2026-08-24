"""Score generated structure with occ_metrics — the eval that could not exist
while coords were pinned to GT.

Usage
  SS_CKPT=runs/s3_ss_t50b/checkpoint-14000 N=32 MODE=i1 \
    python scripts/_tmp/eval_occ_iou.py

MODE  i1   single-image conditioning (the flagship)
      t    text conditioning (multi-modal: IoU against ONE ground truth is a
           weak signal here by construction, so read it as a floor check, not
           as a quality score)

Reference is dec(z_gt) through the SAME frozen decoder the generated latent goes
through, so the VAE's own reconstruction loss cancels and the ceiling is 1.0 by
construction (verified: 1.0000 on 32 val200 assets). Trivial baselines are
printed alongside, because an IoU without them is unreadable — on this dataset a
filled bounding box already scores 0.12.

Sampler params come from geotex_sampler.SS_PARAMS, i.e. the released TRELLIS.2-4B
values, so the number is comparable to the ones on record (_flow16_ref.py header:
AR rollout 0.089, flow head 0.403, both IoU@64).
"""
import json
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")

import numpy as np
import torch

from trellis2_blip3o import _paths  # noqa: F401
from trellis2 import models as t2models
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2_blip3o.geotex_sampler import SS_PARAMS
from trellis2_blip3o.flow_heads import build_unified_cond
from trellis2_blip3o.live_cond_batch import TrainCondEncoder, prep_i1, prep_t
from trellis2_blip3o.occ_metrics import (aggregate, occ64_from_latent, occ_metrics,
                                         trivial_baselines)
from trellis2_blip3o.tr2_modules import SS_FLOW_CONFIG_PATH, load_norm_stats
from scripts.export_glb_fullchain import SSDEC, load_ss_flow
from scripts.eval_fusion_v22 import good_view_b  # noqa: F401 (kept for parity)


def good_view_idx(sha: str) -> int:
    """The INTEGER view index behind eval_fusion_v22.good_view_b, which returns a
    filename. prep_i1 takes an index, so deriving it from the same formula keeps
    this eval on the exact view every other eval in the repo uses."""
    return 5 + (int(sha[:8], 16) + 3) % 7

SS_CKPT = os.environ.get("SS_CKPT", "runs/s3_ss_t50b/checkpoint-14000")
N = int(os.environ.get("N", "32"))
MODE = os.environ.get("MODE", "i1")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
MANI = os.environ.get("MANI", "manifests/splits/val200_capT.jsonl")
USE_EMA = os.environ.get("USE_EMA", "1") == "1"

print(f"[occ] ckpt={SS_CKPT} ema={USE_EMA} mode={MODE} n={N} seeds={SEEDS}", flush=True)
print(f"[occ] sampler={SS_PARAMS}", flush=True)

flow, conn, dve = load_ss_flow(SS_CKPT, use_ema=USE_EMA)
sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
ssdec = t2models.from_pretrained(SSDEC).cuda().eval()
enc = TrainCondEncoder()
# The released SS config has no normalization key by upstream design, so this is
# the identity; keeping the branch means a future upstream change fails loudly
# instead of silently shifting every latent.
_nrm = load_norm_stats(SS_FLOW_CONFIG_PATH, "normalization")
NM = _nrm["mean"].cuda().view(1, -1, 1, 1, 1) if _nrm else 0.0
NS = _nrm["std"].cuda().view(1, -1, 1, 1, 1) if _nrm else 1.0
print(f"[occ] ss normalization: {'ACTIVE' if _nrm else 'none (raw latent)'}", flush=True)


def cond_pair(rec):
    """(cond, uncond) through build_unified_cond — the TRAINING builder.

    Using connector(cond_hidden) alone would silently run the qwen-only arm: the
    s3 SS runs are fuse_dino=True, so their cond is cat([dino_segment,
    qwen_segment]) with a per-view embedding on the dino side, and dropping the
    dino half costs ~1029 of ~2053 tokens. It is not a crash, it is a much worse
    number — measured 0.156 iou64 qwen-only vs the fusion arm below.

    uncond forces the SAME masks CFG uses at inference: qwen zeroed BEFORE the
    connector (connector(0) is the model's learned uncond; zeros after it is an
    unconditional the model never saw) and the dino segment masked out. Forcing
    them through ext_drops rather than probabilities also keeps this
    deterministic.
    """
    h = rec["cond_hidden"].float().cuda()[None]
    km = rec["cond_keep_mask"].cuda().bool()[None]
    dh = rec["dino_hidden"].float().cuda()[None]
    dkm = rec["dino_keep_mask"].cuda().bool()[None]
    dvi = rec["dino_view_ids"].cuda()[None]
    qvi = rec.get("qwen_view_ids")
    qvi = qvi.cuda()[None] if qvi is not None else None
    kw = dict(dino_hidden=dh, dino_key_mask=dkm, dino_view_ids=dvi,
              qwen_view_ids=qvi, dino_view_embed=dve, cond_max_length=10240)
    one = torch.ones(1, dtype=torch.bool, device="cuda")
    zero = torch.zeros(1, dtype=torch.bool, device="cuda")
    c, kc, _, _ = build_unified_cond(conn, h, km, mask_drop_prob=0.0,
                                     dino_drop_prob=0.0, qwen_drop_prob=0.0,
                                     ext_drops=(zero, zero, zero), **kw)
    u, ku, _, _ = build_unified_cond(conn, h, km, mask_drop_prob=0.0,
                                     dino_drop_prob=0.0, qwen_drop_prob=0.0,
                                     ext_drops=(one, one, zero), **kw)
    return c[0][kc[0]][None], u[0][ku[0]][None]


rows = [json.loads(l) for l in open(MANI)]
rows = [r for r in rows if os.path.exists(r["ss_latent_64"])][:N]

# SHUFFLE_COND=1 pairs each asset's GT with the NEXT asset's conditioning. If the
# score barely moves, the model is not reading the image at all and every number
# here is measuring an unconditional prior — the occupancy analogue of the
# cond-sensitivity probe, and the one control that separates "the model is weak"
# from "the conditioning never arrived". Conds are therefore encoded up front.
SHUF = os.environ.get("SHUFFLE_COND", "0") == "1"

recs, gts, base_rows = [], [], []
for r in rows:
    sha = r["sha256"]
    gts.append(occ64_from_latent(ssdec, torch.from_numpy(
        np.load(r["ss_latent_64"])["z"]).float().cuda()))
    base_rows.append(trivial_baselines(gts[-1]))
    if MODE == "t":
        caps = r.get("captions") or []
        cap = caps[0] if caps else "a 3D object"
        if isinstance(cap, dict):
            cap = cap.get("holistic") or next(iter(cap.values()))
        recs.append(enc.encode([prep_t(sha, 0, cap)])[0])
    else:
        recs.append(enc.encode([prep_i1(r["renders_dir"], good_view_idx(sha))])[0])
print(f"[occ] encoded {len(recs)} conds"
      f"{'  (SHUFFLED — control arm)' if SHUF else ''}", flush=True)

per_asset = []
for i in range(len(rows)):
    c, u = cond_pair(recs[(i + 1) % len(recs)] if SHUF else recs[i])
    for sd in SEEDS:
        g = torch.Generator(device="cuda").manual_seed(sd)
        noise = torch.randn(1, flow.in_channels, *[flow.resolution] * 3,
                            generator=g, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = sampler.sample(flow, noise, cond=c, neg_cond=u, verbose=False,
                               **SS_PARAMS).samples
        gen64 = occ64_from_latent(ssdec, z.float() * NS + NM)
        m = occ_metrics(gen64, gts[i])
        m["seed"] = float(sd)
        per_asset.append(m)
    if (i + 1) % 8 == 0:
        cur = aggregate(per_asset, ["iou64"])["iou64"]
        print(f"  [{i+1}/{len(rows)}] running iou64 mean {cur['mean']:.4f}", flush=True)

KEYS = ["iou64", "prec64", "rec64", "prec64_t1", "rec64_t1", "chamfer64",
        "vox_ratio64", "iou32", "prec32", "rec32", "empty", "over_cap"]
A = aggregate(per_asset, KEYS)
Bl = aggregate(base_rows, ["iou_box", "iou_sphere"])

print(f"\n═══ occupancy, n={len(per_asset)} ({len(rows)} assets x {len(SEEDS)} seeds), "
      f"mode={MODE} ═══")
print(f"{'metric':<14}{'mean':>10}{'median':>10}{'p10':>10}")
for k in KEYS:
    v = A[k]
    inf = f"   inf {v['inf_frac']:.0%}" if "inf_frac" in v else ""
    print(f"{k:<14}{v['mean']:>10.4f}{v['median']:>10.4f}{v['p10']:>10.4f}{inf}")
print(f"\ntrivial floors (same GT):  filled bbox {Bl['iou_box']['mean']:.4f}   "
      f"equal-volume sphere {Bl['iou_sphere']['mean']:.4f}")
print(f"on record (IoU@64):        AR rollout 0.089   flow head 0.403")
print("\nread prec/rec together with iou64: a model that is merely one voxel too "
      "fat scores iou64 ~0.39 with rec ~1.0, which iou alone cannot distinguish "
      "from a genuinely mediocre model.")
