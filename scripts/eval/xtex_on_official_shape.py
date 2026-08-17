#!/usr/bin/env python
"""CROSS TEST: our texture stream on TRELLIS.2's shape latent.

The question this settles. Our texture is excellent when conditioned on the GT
shape latent and mediocre when conditioned on a shape latent OUR model sampled.
Two explanations fit that:

  (A) our geometry is simply worse, and the texture faithfully paints whatever
      surface it is handed;
  (B) our texture stream only works on ENCODER-produced latents and degrades on
      any SAMPLED latent — a train/inference distribution mismatch, i.e. an
      alignment bug we could fix without a bigger model.

Feeding it TRELLIS.2's SAMPLED shape latent separates them. Official latents are
sampled (not encoder output) but come from a much stronger model:

  * texture stays good  -> (A). Our tex is robust to sampled latents; the gap is
                           purely geometry quality.
  * texture degrades    -> (B). Our tex is tied to encoder statistics and the
                           whole cascade/joint number is depressed by a fixable
                           mismatch, not by geometry.

Preconditions verified before writing this (do not assume them again):
  * pipeline.sample_shape_slat returns the latent DENORMALIZED
    (trellis2_image_to_3d.py:271-274, `slat * std + mean`), the same space as
    our npz `feats`.
  * the two normalizations are the same numbers: our TEX_SLAT_CONFIG_PATH vs the
    released pipeline.json agree to 2.3e-7 on shape and 1.2e-7 on tex.

Usage (compute node):
  EVAL_MANI=$PWD/manifests/v5_matclean_cached_heldout.jsonl EVAL_ANY=1 \
  python scripts/xtex_on_official_shape.py --ckpt <ckpt> --n 8 \
      --render runs/xtex_official_shape.png
"""
import argparse
import os
import sys

import numpy as np
import torch

# repo root is THREE levels up now (scripts/eval/x.py, scripts/data/x.py);
# it was two when these lived directly under scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from trellis2_blip3o import _paths  # noqa: F401

from scripts.eval_fusion_v22 import build_cond, good_view_b, cam_from_transforms, input_image
import scripts.eval_fusion_v22 as EV
from scripts.eval_geotex_g1g2 import pick_assets
from scripts.eval_tex_v22 import render_textured, HDR
from scripts.eval.eval_overfit10 import load_scratch
from trellis2_blip3o.tr2_modules import (load_norm_stats, TEX_SLAT_CONFIG_PATH,
                                         build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen)
from trellis2_blip3o.geotex_sampler import GeoTexSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", default="runs/xtex_official_shape.png")
    a = ap.parse_args()

    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from trellis2.renderers import EnvMap
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    envmap = EnvMap(torch.tensor(cv2.cvtColor(
        cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)).cuda())
    sdec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tdec = build_sc_vae_tex_decoder_frozen().cuda().eval()

    m, conn, dve = load_scratch(a.ckpt)
    smp = GeoTexSampler(m)
    sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    sm, ssd = sn["mean"].cuda(), sn["std"].cuda()
    tm, tsd = tn["mean"].cuda(), tn["std"].cuda()

    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()

    CELL, HDRH = 512, 96
    cols = ["input image", "GT (both)", "our tex | GT shape",
            "our tex | OFFICIAL shape\n(the test)",
            "OFFICIAL tex | OFFICIAL shape", "ours joint (both ours)"]
    recs = pick_assets(a.n)
    grid = Image.new("RGB", (CELL * len(cols), HDRH + CELL * len(recs)), (250, 250, 250))
    font = ImageFont.truetype(
        "/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 28)
    d = ImageDraw.Draw(grid)
    for c, lab in enumerate(cols):
        d.text((c * CELL + 10, 10), lab, fill=(10, 10, 10), font=font)

    f = lambda z: z.feats if hasattr(z, "feats") else z
    gstd = lambda z: float(f(z).std())
    print(f"{'asset':14} {'gt vox':>7} {'off vox':>8} {'std ours/GT':>12} "
          f"{'std off/GT':>11}", flush=True)

    for ri, (r, ed) in enumerate(recs):
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        rdir = r.get("renders_dir") or r.get("renders_cond_dir")
        extr, intr = cam_from_transforms(rdir)
        gt_s, gt_t = np.load(r["shape_latent_512"]), np.load(r["pbr_latent_512"])
        cx = torch.from_numpy(gt_s["coords"]).int()
        coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
        s_raw = torch.from_numpy(gt_s["feats"]).float().cuda()
        t_raw = torch.from_numpy(gt_t["feats"]).float().cuda()
        c, u = build_cond(conn, dve, ed)

        # --- official: sample its own structure + shape + tex ------------------
        img = Image.open(os.path.join(rdir, EV._VIEW_FILE)).convert("RGBA")
        mesh_off, (s_off_raw, t_off_raw, res) = pipe.run(
            img, seed=a.seed, pipeline_type="512", return_latent=True)
        assert res == 512, f"official returned res={res}, our decoder is pinned to 512"
        c_off = s_off_raw.coords.int()
        # OUR texture, conditioned on THEIR shape latent (renormalized with the
        # stats just verified identical)
        s_off_n = (s_off_raw.feats.float().cuda() - sm) / ssd
        x_tm_off = smp.sample_tex_given_mesh(c_off, s_off_n, c, c, u, seed=a.seed)

        # --- ours on GT coords -------------------------------------------------
        x_tm_gt = smp.sample_tex_given_mesh(coords, (s_raw - sm) / ssd, c, c, u, seed=a.seed)
        js, jt = smp.sample_joint(coords, c, u, c, u, alpha=32, seed=a.seed)

        R = lambda cd, sf, tf: render_textured(sdec, tdec, cd, sf, tf, extr, intr, envmap)
        row = [input_image(rdir).resize((CELL, CELL)),
               R(coords, s_raw, t_raw),
               R(coords, s_raw, f(x_tm_gt) * tsd + tm),
               R(c_off, s_off_raw.feats.float().cuda(), f(x_tm_off) * tsd + tm),
               R(c_off, s_off_raw.feats.float().cuda(), t_off_raw.feats.float().cuda()),
               R(coords, f(js) * ssd + sm, f(jt) * tsd + tm)]
        for ci, im in enumerate(row):
            grid.paste(im.resize((CELL, CELL)), (ci * CELL, HDRH + ri * CELL))

        rt = float(((t_raw - tm) / tsd).std())
        print(f"{sha[:12]:14} {cx.shape[0]:7d} {c_off.shape[0]:8d} "
              f"{gstd(x_tm_off)/rt:11.0%} "
              f"{float(((t_off_raw.feats.float().cuda()-tm)/tsd).std())/rt:10.0%}",
              flush=True)
        torch.cuda.empty_cache()

    grid.save(a.render)
    print(f"\n[xtex] grid -> {a.render}")


if __name__ == "__main__":
    main()
