"""Multi-seed tex|mesh grid: one GT geometry, several texture samples.
Columns: input | GT | specialist(seed0) | unified seed0 | seed1 | seed2.
Reuses the certified GeoTexSampler tex_given_mesh path (geometry locked)."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis2_blip3o import _paths  # noqa: F401

import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import build_cond, good_view_b, cam_from_transforms, input_image
from scripts.eval_tex_v22 import load_tex_flow, sample_tex, render_textured, HDR
from scripts.eval_geotex_g1g2 import pick_assets, load_geotex, TEX_SPEC_CKPT
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen,
                                         load_norm_stats, TEX_SLAT_CONFIG_PATH)
from trellis2_blip3o.geotex_sampler import GeoTexSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/geotex_s1_smoke/checkpoint-3")
    ap.add_argument("--coupling", default="union")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2")
    # None = released TRELLIS.2-4B per-stream params.
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--out", default="runs/cache_logs/eval_geotex")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]

    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from trellis2.renderers import EnvMap

    recs = pick_assets(a.n)
    uni, conn_g, conn_x, dve = load_geotex(a.ckpt, a.coupling)
    smp = GeoTexSampler(uni, steps=a.steps, cfg=a.cfg)
    tflow, tconn, tdve = load_tex_flow(TEX_SPEC_CKPT)

    sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    sm, ssd = sn["mean"].cuda(), sn["std"].cuda()
    tm, tsd = tn["mean"].cuda(), tn["std"].cuda()
    shape_dec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tex_dec = build_sc_vae_tex_decoder_frozen().cuda().eval()
    hdr = cv2.cvtColor(cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(hdr).cuda())
    font = ImageFont.truetype(
        "/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 40)

    cell, hdrh = 640, 64
    cols = ["input", "GT", f"specialist s{seeds[0]}"] + [f"unified s{s}" for s in seeds]
    grid = Image.new("RGB", (cell * len(cols), hdrh + cell * len(recs)), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    for c, lab in enumerate(cols):
        d.text((c * cell + 12, 12), lab, fill=(10, 10, 10), font=font)

    for ri, (r, ed) in enumerate(recs):
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        gt_s = np.load(r["shape_latent_512"]); gt_t = np.load(r["pbr_latent_512"])
        cx = torch.from_numpy(gt_s["coords"]).int()
        coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
        shape_raw = torch.from_numpy(gt_s["feats"]).float().cuda()
        shape_norm = (shape_raw - sm) / ssd
        c_s, _ = build_cond(conn_g, dve, ed)
        c_x, u_x = build_cond(conn_x, dve, ed)
        c_xs, u_xs = build_cond(tconn, tdve, ed)
        rdir = r.get("renders_dir") or r.get("renders_cond_dir")
        extr, intr = cam_from_transforms(rdir)

        row = [input_image(rdir),
               render_textured(shape_dec, tex_dec, coords, shape_raw,
                               torch.from_numpy(gt_t["feats"]).float().cuda(),
                               extr, intr, envmap)]
        t_spec = sample_tex(tflow, c_xs, u_xs, coords, shape_norm,
                            steps=a.steps, cfg=a.cfg, seed=seeds[0])
        row.append(render_textured(shape_dec, tex_dec, coords, shape_raw,
                                   t_spec * tsd + tm, extr, intr, envmap))
        for s in seeds:
            t_u = smp.sample_tex_given_mesh(coords, shape_norm, c_s, c_x, u_x, seed=s)
            row.append(render_textured(shape_dec, tex_dec, coords, shape_raw,
                                       t_u * tsd + tm, extr, intr, envmap))
        for c, im in enumerate(row):
            grid.paste(im.resize((cell, cell)), (c * cell, hdrh + ri * cell))
        print(f"[multiseed] {sha[:12]} done")

    gp = os.path.join(a.out,
                      f"g2ms_{os.path.basename(os.path.dirname(a.ckpt))}.png")
    grid.save(gp)
    print(f"[multiseed] grid: {gp}")


if __name__ == "__main__":
    main()
