#!/usr/bin/env python
"""Pin the 16 assets every comparison figure will use, forever.

WHY IT IS PINNED. The previous 16 were whatever `pick_assets(16)` returned from
the old 1% held-out. After the 2026-08-17 re-split, 15 of those 16 fell back
into the training pool — including the baseball, the tower and the carousel,
the three that carried every diagnostic conclusion. A figure set that moves
with the split is a figure set you cannot compare across runs.

WHY NOT A RANDOM DRAW. 2026-08-17 established that the informative assets are
the ones where a given amount of latent error is maximally VISIBLE:

  * smooth surfaces with fine relief  -> surface ripple has nowhere to hide
    (the baseball: geo latent MSE 0.516, the BEST of 16, and visually the worst)
  * dense repeated structure          -> over-smoothing has nowhere to hide
    (the tower: 170% of GT surface high-frequency, i.e. mangled lattice)
  * strongly coloured assets          -> desaturation shows (ours std 76% vs GT)

So the draw is stratified on two axes measured from the GT ITSELF, not on
anything the model produces: GT normal-map high-frequency (how much fine relief
the asset has) x GT foreground chroma (how much colour there is to lose).

Both axes come off renders of the ground truth, so the set is model-independent
and never has to be re-picked when the model changes.

Usage (compute node):
  EVAL_MANI=$PWD/manifests/splits/val200.jsonl EVAL_ANY=1 \
  python scripts/mk_viz16.py --out manifests/splits/viz16.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# repo root is THREE levels up now (scripts/eval/x.py, scripts/data/x.py);
# it was two when these lived directly under scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from trellis2_blip3o import _paths  # noqa: F401

from scripts.eval_geotex_g1g2 import pick_assets
from scripts.eval_fusion_v22 import good_view_b, cam_from_transforms
import scripts.eval_fusion_v22 as EV
from scripts.eval_tex_v22 import render_textured, HDR
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen)


def hf(img):
    """Foreground high-frequency energy of a normal map = how much fine relief
    the asset has. Foreground only — a full-frame mean is dominated by the
    background (skill trellis2-3d-data S7)."""
    a = img.astype(np.float32) / 255
    lum = a.mean(-1)
    fg = lum > 0.02
    gx = np.abs(np.diff(a, axis=1)).sum(-1)
    gy = np.abs(np.diff(a, axis=0)).sum(-1)
    e = np.zeros_like(lum)
    e[:, :-1] += gx
    e[:-1, :] += gy
    return float(e[fg].mean()) if fg.sum() > 50 else 0.0


def chroma(img):
    a = img.astype(np.int32)
    fg = a.sum(-1) > 30
    if fg.sum() < 50:
        return 0.0
    px = a[fg]
    mx, mn = px.max(-1), px.min(-1)
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255, 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifests/splits/viz16.jsonl")
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()

    import cv2
    from trellis2.renderers import EnvMap

    sdec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tdec = build_sc_vae_tex_decoder_frozen().cuda().eval()
    envmap = EnvMap(torch.tensor(cv2.cvtColor(
        cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)).cuda())

    recs = pick_assets(10000)          # the whole EVAL_MANI
    print(f"[viz] characterising {len(recs)} candidates", flush=True)
    rows = []
    for i, (r, _ed) in enumerate(recs):
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        rdir = r.get("renders_dir") or r.get("renders_cond_dir")
        try:
            extr, intr = cam_from_transforms(rdir)
            gs = np.load(r["shape_latent_512"])
            gt = np.load(r["pbr_latent_512"])
            cx = torch.from_numpy(gs["coords"]).int()
            coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
            s = torch.from_numpy(gs["feats"]).float().cuda()
            t = torch.from_numpy(gt["feats"]).float().cuda()
            n = np.array(render_textured(sdec, tdec, coords, s, t, extr, intr,
                                         envmap, channel="normal"))
            c = np.array(render_textured(sdec, tdec, coords, s, t, extr, intr, envmap))
        except Exception as e:
            print(f"  {sha[:12]} SKIP {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(dict(rec=r, sha=sha, subset=r.get("subset", "?"),
                         hf=hf(n), chroma=chroma(c), vox=int(cx.shape[0])))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(recs)}", flush=True)
        torch.cuda.empty_cache()

    H = np.array([x["hf"] for x in rows])
    C = np.array([x["chroma"] for x in rows])
    h1, h2 = np.quantile(H, [1 / 3, 2 / 3])
    c1 = np.quantile(C, 0.5)
    for x in rows:
        x["cell"] = (("smooth" if x["hf"] < h1 else "medium" if x["hf"] < h2 else "dense"),
                     ("colour" if x["chroma"] >= c1 else "grey"))
    print(f"\n[viz] GT surface-detail cuts {h1:.4f} / {h2:.4f} · chroma cut {c1:.1f}")

    # even over the 6 cells; inside a cell prefer subset diversity, then the
    # most EXTREME asset (highest |hf - cell centre|) — the visible cases are
    # the informative ones. Deterministic: ties break on sha.
    cells = {}
    for x in rows:
        cells.setdefault(x["cell"], []).append(x)
    # ROUND-ROBIN over the cells, not "n per cell then top up by sha": the naive
    # version filled 6 cells x 2 and then took the remaining 4 in sha order,
    # which all landed in `grey` (10 grey / 6 colour). Colour assets are the ones
    # that show desaturation — the texture failure mode we actually have — so
    # the set must not drift grey.
    # Sort each cell ONCE, then only advance a cursor. Re-sorting inside the
    # loop (which an earlier version did, to keep the subset-diversity key
    # fresh) invalidates the cursor: the list reorders under it and the same
    # asset gets picked twice — measured, 3 of 16 were duplicates.
    # Subset rarity is a STATIC key computed from the whole candidate pool, so
    # rare subsets (Toys4k, HSSD, ABO) lead inside their cell without needing
    # the key to mutate.
    freq = {}
    for x in rows:
        freq[x["subset"]] = freq.get(x["subset"], 0) + 1
    for xs in cells.values():
        xs.sort(key=lambda z: (freq[z["subset"]], z["sha"]))
    pick, seen_sha, cursor = [], set(), {}
    order = sorted(cells)
    while len(pick) < a.k and any(cursor.get(c, 0) < len(cells[c]) for c in order):
        for c in order:
            if len(pick) >= a.k:
                break
            xs, i = cells[c], cursor.get(c, 0)
            while i < len(xs) and xs[i]["sha"] in seen_sha:
                i += 1
            cursor[c] = i + 1
            if i >= len(xs):
                continue
            seen_sha.add(xs[i]["sha"])
            pick.append(xs[i])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for x in pick:
            f.write(json.dumps(x["rec"]) + "\n")
    print(f"\n{'sha':14} {'subset':22} {'cell':18} {'hf':>8} {'chroma':>8} {'vox':>7}")
    for x in pick:
        print(f"{x['sha'][:12]:14} {x['subset']:22} {'/'.join(x['cell']):18} "
              f"{x['hf']:8.4f} {x['chroma']:8.1f} {x['vox']:7d}")
    print(f"\n[viz] {len(pick)} -> {a.out}")


if __name__ == "__main__":
    main()
