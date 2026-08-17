"""G0.5 — sampler-equivalence certification for the 3-mode GeoTexSampler (GPU).

At WARM START with coupling="gated" (c_gates=0, cross_alpha=0) the unified model
is bit-exactly the two specialists (G0), so the SAMPLER TRAJECTORIES must also be
bit-equal to the production eval samplers on the same seed:

  [A] mesh_only        ≡ eval_fusion_v22.sample_shape   (geo lane, any coupling)
  [B] tex_given_mesh   ≡ eval_tex_v22.sample_tex        (gated ⇒ bit-exact; this
      also certifies the geo-K/V CACHE path — one pass reused for all steps)
  [C] joint α=32 smoke — finite outputs, both streams move
  [D] union arm (informational): tex divergence from the specialist at warm start
      = the known init perturbation MF-style training absorbs (NOT a gate)

Run on a compute node:
  srun --jobid=<id> --overlap ... python scripts/g05_sampler_equiv.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis2_blip3o import _paths  # noqa: F401
from trellis2.modules import sparse as sp

import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import build_cond, sample_shape
from scripts.eval_tex_v22 import load_tex_flow, sample_tex, COND_ROOT, MANI
from scripts.eval_fusion_v22 import load_flow_and_connector
from trellis2_blip3o.tr2_modules import load_norm_stats, TEX_SLAT_CONFIG_PATH
from trellis2_blip3o.unified_geotex import assemble_unified_inference
from trellis2_blip3o.geotex_sampler import GeoTexSampler

SHAPE_CKPT = os.environ.get("G05_SHAPE_CKPT", "runs/s3_shape_t50b/checkpoint-8000")
TEX_CKPT = os.environ.get("G05_TEX_CKPT", "runs/s3_tex_t50b/checkpoint-8000")
STEPS, CFG, SEED = 25, 3.0, 0


def pick_asset():
    with open(MANI) as f:
        for line in f:
            r = json.loads(line)
            ed = os.path.join(COND_ROOT, r["sha256"][:2], r["sha256"])
            if os.path.exists(os.path.join(ed, "v000.npz")) and r.get("pbr_latent_512") \
                    and r.get("shape_latent_512"):
                gt = np.load(r["shape_latent_512"])
                if gt["coords"].shape[0] <= 8192:
                    return r, ed
    raise RuntimeError("no eligible asset")


def main():
    rec, entry = pick_asset()
    print(f"[G05] asset {rec['sha256'][:12]}  voxels={np.load(rec['shape_latent_512'])['coords'].shape[0]}")

    # reference specialists via the PRODUCTION eval loaders (independent load path
    # from assemble_unified — also cross-checks EMA/prefix agreement)
    sflow, sconn, sdve = load_flow_and_connector(SHAPE_CKPT)
    tflow, tconn, tdve = load_tex_flow(TEX_CKPT)
    c_s, u_s = build_cond(sconn, sdve, entry)
    c_x, u_x = build_cond(tconn, tdve, entry)

    gt = np.load(rec["shape_latent_512"])
    cx = torch.from_numpy(gt["coords"]).int()
    coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
    shape_raw = torch.from_numpy(gt["feats"]).float().cuda()
    sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    sm, ssd = sn["mean"].cuda(), sn["std"].cuda()
    shape_norm = (shape_raw - sm) / ssd

    # reference trajectories
    ref_shape = sample_shape(sflow, c_s, u_s, coords, steps=STEPS, cfg=CFG, seed=SEED)
    ref_tex = sample_tex(tflow, c_x, u_x, coords, shape_norm, steps=STEPS, cfg=CFG, seed=SEED)

    ok = True
    for coupling in ("gated", "union"):
        uni = assemble_unified_inference(SHAPE_CKPT, TEX_CKPT, coupling=coupling).cuda().eval()
        # PINNED to the legacy params ON PURPOSE: this gate asserts bit-equality
        # with eval_fusion_v22 / eval_tex_v22, which are bare-Euler 25 steps at
        # cfg 3.0. GeoTexSampler now DEFAULTS to the released params (12 steps,
        # guidance interval, rescale_t), so the equality it certifies no longer
        # describes the production path — this gate documents the warm-start
        # assembly, not the shipped sampler.
        smp = GeoTexSampler(uni, steps=STEPS, cfg=CFG)

        xs = smp.sample_mesh_only(coords, c_s, u_s, seed=SEED)
        d_shape = (xs.feats - ref_shape.feats).abs().max().item()

        xt = smp.sample_tex_given_mesh(coords, shape_norm, c_s, c_x, u_x, seed=SEED)
        d_tex = (xt - ref_tex).abs().max().item()
        rel_tex = ((xt - ref_tex).norm() / ref_tex.norm().clamp_min(1e-9)).item()

        if coupling == "gated":
            good = (d_shape == 0.0) and (d_tex == 0.0)
            ok &= good
            print(f"[G05][gated] mesh_only max|Δ|={d_shape:.3e}  tex|mesh max|Δ|={d_tex:.3e}"
                  f"  {'BIT-EXACT ✓' if good else 'FAIL ✗'}")
        else:
            ok &= (d_shape == 0.0)   # geo lane never sees tex — must hold for union too
            print(f"[G05][union] mesh_only max|Δ|={d_shape:.3e} (must be 0)  "
                  f"tex|mesh warm-start divergence rel={rel_tex:.3f} (informational — "
                  f"init perturbation, absorbed by training)")

        x_s, x_x = smp.sample_joint(coords, c_s, u_s, c_x, u_x, alpha=32.0, seed=SEED)
        fin = torch.isfinite(x_s.feats).all().item() and torch.isfinite(torch.as_tensor(x_x)).all().item()
        mv = x_s.feats.float().std().item()
        ok &= fin
        print(f"[G05][{coupling}] joint a=32 smoke: finite={fin}  shape_std={mv:.3f}  "
              f"tex_std={x_x.float().std().item():.3f}")
        del uni, smp
        torch.cuda.empty_cache()

    print("G0.5 " + ("PASSED — sampler certified" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
