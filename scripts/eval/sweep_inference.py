#!/usr/bin/env python
"""Inference-side ablations on a FIXED checkpoint — no training involved.

Three knobs we inherited without ever validating them on our own model:

  A1  alpha        the joint schedule's warp. At alpha=32 the texture stream sits
                   at t_x~1 for essentially the whole geometry trajectory, so the
                   cross-modal coupling is unavailable exactly when geometry is
                   being formed. Measured on checkpoint-60000: handing geometry
                   the GT texture improves its single-step prediction by 59.5%,
                   while the entire image conditioning is worth 7.7%. If that
                   coupling can be cashed in at all, LOWERING alpha is how.
  A2  geo cfg      TRELLIS.2 ships 7.5. Our cond-uncond direction is near zero
                   (std 0.043-0.086) because the geometry stream barely reads the
                   image, so 7.5 amplifies noise: on the baseball it moved
                   surface high-frequency from 73% of GT to 181% and texture MSE
                   from 1.158 to 1.424.
  A3  refine_tex   on by default (MF runner.py:311-326), never A/B'd here.

Reported per setting: geometry and texture latent MSE, std/GT (a collapsed
sampler wins on MSE while looking washed out, so MSE alone is not a score), and
surf_dev = |normal-map high-frequency / GT - 1|, the one metric that agreed with
the eye in the 2026-08-17 audit. See skill `geotex-model-eval`.

Usage (compute node):
  EVAL_MANI=$PWD/manifests/splits/val200.jsonl EVAL_ANY=1 \
  python scripts/eval/sweep_inference.py --ckpt <ckpt> --n 32 --sweep alpha
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from trellis2_blip3o import _paths  # noqa: F401

from scripts.eval_geotex_g1g2 import pick_assets
from scripts.eval_fusion_v22 import good_view_b, cam_from_transforms, build_cond
import scripts.eval_fusion_v22 as EV
from scripts.eval_tex_v22 import render_textured, HDR
from scripts.eval.eval_overfit10 import load_scratch
from trellis2_blip3o.tr2_modules import (load_norm_stats, TEX_SLAT_CONFIG_PATH,
                                         build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen)
from trellis2_blip3o.geotex_sampler import GeoTexSampler


def hf(img):
    """Foreground high-frequency energy of a normal map. Foreground only — a
    full-frame mean is dominated by the background (skill trellis2-3d-data S7)."""
    a = img.astype(np.float32) / 255
    lum = a.mean(-1)
    fg = lum > 0.02
    gx = np.abs(np.diff(a, axis=1)).sum(-1)
    gy = np.abs(np.diff(a, axis=0)).sum(-1)
    e = np.zeros_like(lum)
    e[:, :-1] += gx
    e[:-1, :] += gy
    return float(e[fg].mean()) if fg.sum() > 50 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--sweep", choices=["alpha", "cfg", "refine"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true",
                    help="also render, which is what surf_dev needs (2x slower)")
    a = ap.parse_args()

    SETTINGS = {"alpha": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
                "cfg": [1.0, 2.0, 3.5, 5.0, 7.5],
                "refine": [False, True]}[a.sweep]

    m, conn, dve = load_scratch(a.ckpt)
    sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    sm, ssd = sn["mean"].cuda(), sn["std"].cuda()
    tm, tsd = tn["mean"].cuda(), tn["std"].cuda()

    dec = None
    if a.render:
        import cv2
        from trellis2.renderers import EnvMap
        dec = (build_sc_vae_shape_decoder_frozen().cuda().eval(),
               build_sc_vae_tex_decoder_frozen().cuda().eval(),
               EnvMap(torch.tensor(cv2.cvtColor(
                   cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)).cuda()))

    recs = pick_assets(a.n)
    f = lambda z: z.feats if hasattr(z, "feats") else z
    acc = {s: {k: [] for k in ("geo", "tex", "gstd", "tstd", "sdev")} for s in SETTINGS}
    print(f"[sweep] {a.sweep} over {SETTINGS} · {len(recs)} assets · {a.ckpt}", flush=True)

    for ri, (r, ed) in enumerate(recs):
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        gs, gt = np.load(r["shape_latent_512"]), np.load(r["pbr_latent_512"])
        cx = torch.from_numpy(gs["coords"]).int()
        coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
        s_raw = torch.from_numpy(gs["feats"]).float().cuda()
        t_raw = torch.from_numpy(gt["feats"]).float().cuda()
        s_gt, t_gt = (s_raw - sm) / ssd, (t_raw - tm) / tsd
        c, u = build_cond(conn, dve, ed)
        base = None
        if dec is not None:
            rdir = r.get("renders_dir") or r.get("renders_cond_dir")
            extr, intr = cam_from_transforms(rdir)
            base = hf(np.array(render_textured(dec[0], dec[1], coords, s_raw, t_raw,
                                               extr, intr, dec[2], channel="normal")))
        for st in SETTINGS:
            smp = GeoTexSampler(m)
            alpha, refine = 32.0, True
            if a.sweep == "alpha":
                alpha = st
            elif a.sweep == "cfg":
                smp.shape_p = {**smp.shape_p, "guidance_strength": st}
            else:
                refine = st
            with torch.no_grad():
                js, jt = smp.sample_joint(coords, c, u, c, u, alpha=alpha,
                                          seed=a.seed, refine_tex=refine)
            acc[st]["geo"].append(float(((f(js) - s_gt) ** 2).mean()))
            acc[st]["tex"].append(float(((f(jt) - t_gt) ** 2).mean()))
            acc[st]["gstd"].append(float(f(js).std()) / float(s_gt.std()))
            acc[st]["tstd"].append(float(f(jt).std()) / float(t_gt.std()))
            if dec is not None:
                nm = np.array(render_textured(dec[0], dec[1], coords,
                                              f(js) * ssd + sm, t_raw, extr, intr,
                                              dec[2], channel="normal"))
                acc[st]["sdev"].append(abs(hf(nm) / base - 1))
            torch.cuda.empty_cache()
        if (ri + 1) % 8 == 0:
            print(f"  {ri + 1}/{len(recs)}", flush=True)

    print(f"\n{a.sweep:>10} {'geo MSE':>9} {'geo std':>8} {'tex MSE':>9} "
          f"{'tex std':>8} {'surf_dev':>9}")
    for st in SETTINGS:
        A = acc[st]
        sd = np.mean(A["sdev"]) if A["sdev"] else float("nan")
        print(f"{str(st):>10} {np.mean(A['geo']):9.4f} {np.mean(A['gstd']):7.0%} "
              f"{np.mean(A['tex']):9.4f} {np.mean(A['tstd']):7.0%} {sd:9.1%}")
    # the argmin the skill's protocol asks for, on the metric that agreed with
    # the eye when one is available, otherwise on texture MSE
    key = "sdev" if acc[SETTINGS[0]]["sdev"] else "tex"
    best = min(SETTINGS, key=lambda s: np.mean(acc[s][key]))
    print(f"\n[sweep] best {a.sweep} by {key}: {best}")


if __name__ == "__main__":
    main()
