"""Contact sheets for eyeballing the content-dedup decisions of build_heldout_clean.py.

A threshold you have not looked through is a guess.  This renders, side by side, the
candidate and the training asset it was matched to, for three bands:
  removed   — pairs above the threshold (should be obvious duplicates)
  nearmiss  — pairs just BELOW the threshold that were KEPT (should be obviously different;
              if they are not, the threshold is too loose)
  random    — kept pairs at a typical similarity, as a control

  python scripts/dedup_spotcheck.py --band removed --n 12 --out /tmp/removed.png
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_heldout_clean import DATA, OUTDIR, renders_dir  # noqa: E402

WORK = f"{OUTDIR}/_work"
FONT = ("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
VIEWS = (2, 6, 9, 13)


def find_renders(sha):
    for sub in os.listdir(DATA):
        p = renders_dir(sub, sha)
        if os.path.isdir(p):
            return p
    return None


def strip(sha, cell):
    d = find_renders(sha)
    im = Image.new("RGB", (cell * len(VIEWS), cell), (255, 255, 255))
    if d is None:
        return im
    for i, v in enumerate(VIEWS):
        p = os.path.join(d, f"{v:03d}.webp")
        if not os.path.isfile(p):
            continue
        a = Image.open(p).convert("RGBA")
        bg = Image.new("RGB", a.size, (255, 255, 255))
        bg.paste(a, mask=a.split()[3])
        im.paste(bg.resize((cell, cell), Image.LANCZOS), (i * cell, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="removed",
                    choices=["removed", "nearmiss", "random", "render_only", "geom_only",
                             "render_borderline", "geom_borderline", "sil_only"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=140)
    a = ap.parse_args()

    rep = json.load(open(f"{WORK}/dedup_report.json"))
    th = rep["thresholds"]
    per = rep["per_candidate"]
    tier = json.load(open(f"{WORK}/cand_tier.json"))

    def score(r):     # how close to being called a duplicate
        return max(r["render_cos"] / th["render_cos"],
                   r.get("sil_cos", -2) / th.get("sil_cos", 1.0),
                   r["geom_cos_exact"] / th["geom_cos"])

    rows = []
    if a.band == "removed":
        c = [(s, r) for s, r in per.items() if r["dup"]]
        c.sort(key=lambda x: -score(x[1]))
    elif a.band == "render_only":
        # the risky class: dropped on appearance alone, geometry says they differ
        c = [(s, r) for s, r in per.items() if r["dup_render"] and not r["dup_geom"]]
        c.sort(key=lambda x: -x[1]["render_cos"])
    elif a.band == "render_borderline":
        c = [(s, r) for s, r in per.items() if r["dup_render"] and not r["dup_geom"]]
        c.sort(key=lambda x: x[1]["render_cos"])          # closest to the threshold
    elif a.band == "geom_only":
        c = [(s, r) for s, r in per.items() if r["dup_geom"] and not r["dup_render"]]
        c.sort(key=lambda x: -x[1]["geom_cos_exact"])
    elif a.band == "geom_borderline":
        # the band between "clearly the same mesh" (1.000) and the threshold
        c = [(s, r) for s, r in per.items()
             if th["geom_cos"] - 0.15 <= r["geom_cos_exact"] < 0.999]
        c.sort(key=lambda x: -x[1]["geom_cos_exact"])
    elif a.band == "sil_only":
        c = [(s, r) for s, r in per.items()
             if r.get("dup_sil") and not r["dup_geom"] and not r["dup_render"]]
        c.sort(key=lambda x: -x[1]["sil_cos"])
    elif a.band == "nearmiss":
        c = [(s, r) for s, r in per.items() if not r["dup"]]
        c.sort(key=lambda x: -score(x[1]))
    else:
        c = [(s, r) for s, r in per.items() if not r["dup"]]
        c.sort(key=lambda x: -score(x[1]))
        c = c[len(c) // 2:]
    for s, r in c[:a.n]:
        m = (r["geom_match_exact"] if r["geom_cos_exact"] >= r["render_cos"]
             else r["render_match"]) or r["render_match"] or r["geom_match_exact"]
        rows.append((s, m, r))

    cell = a.cell
    W = cell * len(VIEWS) * 2 + 40
    hdr = 34
    H = hdr + len(rows) * (cell + 30)
    out = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(out)
    f1 = ImageFont.truetype(FONT, 18)
    f2 = ImageFont.truetype(FONT, 14)
    d.text((8, 8), f"band={a.band}  render>={th['render_cos']} sil>={th.get('sil_cos','-')} "
                   f"geom>={th['geom_cos']}   LEFT=held-out candidate   "
                   f"RIGHT=nearest TRAINING asset", fill=(0, 0, 0), font=f1)
    for i, (s, m, r) in enumerate(rows):
        y = hdr + i * (cell + 30)
        out.paste(strip(s, cell), (0, y))
        if m:
            out.paste(strip(m, cell), (cell * len(VIEWS) + 40, y))
        d.text((4, y + cell + 4),
               f"{s[:12]} tier{tier.get(s,'?')}  render={r['render_cos']:.3f} "
               f"sil={r.get('sil_cos',float('nan')):.3f} geom={r['geom_cos_exact']:.3f} "
               f"-> {str(m)[:12]}   dup={r['dup']} (r={r['dup_render']} "
               f"s={r.get('dup_sil')} g={r['dup_geom']})",
               fill=(20, 20, 20), font=f2)
    out.save(a.out)
    print(f"wrote {a.out}  rows={len(rows)}")


if __name__ == "__main__":
    main()
