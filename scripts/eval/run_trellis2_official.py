#!/usr/bin/env python
"""Run the RELEASED microsoft/TRELLIS.2-4B on the same held-out assets our eval
uses, and cache one render per asset so eval_overfit10.py can paste it in as the
rightmost reference column.

Everything that is not the generator is shared with our eval on purpose:

  * asset list      pick_assets()          (scripts/eval_geotex_g1g2)
  * view choice     good_view_b(sha)       (scripts/eval_fusion_v22)
  * camera          cam_from_transforms()  (same extrinsics/intrinsics)
  * envmap + render t2render.render_frames on the SAME HDR

so the only thing that differs between our column and this one is what produced
the mesh.

TWO differences that must be stated whenever this column is shown:

  1. WE ARE GIVEN THE GT VOXEL OCCUPANCY. eval_overfit10.py:233 takes `coords`
     straight out of the ground-truth shape latent; our model only fills
     features onto voxels somebody else decided were occupied. The released
     pipeline generates its own sparse structure from the image (the 64^3
     ss_flow stage) and then has to live with it. This is a large advantage for
     us and it is invisible in any latent-MSE number, because MSE is only ever
     computed on the coords we were handed.
  2. RESOLUTION. Our latents are 512; the released default is `1024_cascade`.
     --pipeline_type 512 gives the matched-resolution comparison, and is the one
     to use when the question is "is our model good", not "is it better than
     what you can download".

The conditioning image is passed as RGBA so the pipeline takes the alpha path
(trellis2_image_to_3d.py:131-142) instead of running BiRefNet. Our renders have
an exact alpha channel; letting rembg re-segment it would inject segmentation
error that has nothing to do with either generator.

Usage (compute node):
  EVAL_MANI=$PWD/manifests/v5_matclean_cached_heldout.jsonl EVAL_ANY=1 \
  python scripts/run_trellis2_official.py --n 16 --pipeline_type 512 \
      --out runs/official_t2/512
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

from scripts.eval_fusion_v22 import good_view_b, cam_from_transforms
import scripts.eval_fusion_v22 as EV
from scripts.eval_geotex_g1g2 import pick_assets
from scripts.eval_tex_v22 import HDR


def cond_rgba(renders_dir):
    """The SAME view our model is conditioned on, with alpha kept.

    eval_fusion_v22.input_image() composites onto white and returns RGB because
    that is what the DINO cache was built from. Here we keep the alpha instead:
    the pipeline's preprocess uses it directly and skips rembg.
    """
    from PIL import Image
    p = os.path.join(renders_dir, EV._VIEW_FILE)
    if not os.path.exists(p):
        c = sorted(f for f in os.listdir(renders_dir) if f.endswith(".webp"))
        p = os.path.join(renders_dir, c[0])
    im = Image.open(p)
    return im.convert("RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True, help="dir for <sha>.png renders")
    ap.add_argument("--pipeline_type", default="512",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"],
                    help="512 = matched to our latent resolution; "
                         "1024_cascade = the released default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render_res", type=int, default=1024)
    a = ap.parse_args()

    import cv2
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils as t2render

    os.makedirs(a.out, exist_ok=True)
    envmap = EnvMap(torch.tensor(cv2.cvtColor(
        cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)).cuda())

    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()

    recs = pick_assets(a.n)
    assert recs, "no assets — set EVAL_MANI and EVAL_ANY=1"
    print(f"[official] {len(recs)} assets · pipeline_type={a.pipeline_type} · "
          f"seed={a.seed} -> {a.out}\n", flush=True)

    for r, _ed in recs:
        sha = r["sha256"]
        dst = os.path.join(a.out, f"{sha}.png")
        if os.path.exists(dst):
            print(f"{sha[:12]}  cached", flush=True)
            continue
        EV._VIEW_FILE = good_view_b(sha)          # same view as our columns
        rdir = r.get("renders_dir") or r.get("renders_cond_dir")
        img = cond_rgba(rdir)
        try:
            mesh = pipe.run(img, seed=a.seed, pipeline_type=a.pipeline_type)[0]
            extr, intr = cam_from_transforms(rdir)
            rd = t2render.render_frames(
                mesh, [extr], [intr],
                {"resolution": a.render_res, "bg_color": (1, 1, 1)}, envmap=envmap)
            key = "shaded" if "shaded" in rd else (
                "color" if "color" in rd else list(rd)[0])
            im = rd[key][0]
            if im.dtype != np.uint8:
                im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(im).save(dst)
            nv = int(mesh.coords.shape[0]) if hasattr(mesh, "coords") else -1
            print(f"{sha[:12]}  ok   vox={nv}", flush=True)
        except Exception as e:
            # never swallow: a missing column must be visibly missing, not a
            # silently reused neighbour
            print(f"{sha[:12]}  FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            torch.cuda.empty_cache()

    done = len([f for f in os.listdir(a.out) if f.endswith(".png")])
    print(f"\n[official] {done}/{len(recs)} rendered -> {a.out}")


if __name__ == "__main__":
    main()
