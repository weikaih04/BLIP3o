"""Characterise the clean held-out set: composition, quality distribution vs TRAINING,
modality completeness.  Run after build_heldout_clean.py --stage manifest and after the
cond caches are built.

  python scripts/heldout_clean_report.py [--json out.json]
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

OUTDIR = "/fsx/home/weikai.huang/3dgen/im_probe/heldout_clean"
WORK = f"{OUTDIR}/_work"
CR = "/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_hoclean"
IMR = "/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_clean_im4r"
FIELDS = ("aesthetic_score", "structural_score", "texture_score",
          "part_complexity", "detail_complexity", "color_richness")


def dist(ds, f):
    v = np.array([d[f] for d in ds if d.get(f) is not None], dtype=np.float64)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{OUTDIR}/heldout_clean.jsonl")
    ap.add_argument("--capt", default=f"{OUTDIR}/heldout_clean_capT.jsonl")
    ap.add_argument("--json", default=f"{OUTDIR}/characterisation.json")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.manifest)]
    caps = {json.loads(l)["sha256"]: json.loads(l)["captions"] for l in open(a.capt)}
    trj = json.load(open(f"{WORK}/train_judge.json"))
    train = list(trj.values())
    out = {"n": len(recs)}

    print(f"=== HELD-OUT SET: {a.manifest} ===")
    print(f"n = {len(recs)}")
    tiers = Counter(r["tier"] for r in recs)
    print(f"tier mix   : {dict(tiers)}")
    subs = Counter(r["subset"] for r in recs)
    trsubs = Counter(d["subset"] for d in train)
    ntr = sum(trsubs.values())
    print(f"\n{'subset':<24} {'held-out':>9} {'%':>7}   {'training %':>11}   skew")
    for s in sorted(set(subs) | set(trsubs)):
        hp = 100 * subs.get(s, 0) / len(recs)
        tp = 100 * trsubs.get(s, 0) / ntr
        print(f"{s:<24} {subs.get(s,0):>9} {hp:>6.1f}%   {tp:>10.1f}%   "
              f"{'x%.2f' % (hp / tp) if tp > 0.05 else '-':>7}")
    out["subset_heldout"] = dict(subs)
    out["subset_train_pct"] = {s: 100 * c / ntr for s, c in trsubs.items()}

    print(f"\n{'field':<20} {'held-out mean':>14} {'sd':>6} | {'train mean':>11} {'sd':>6} "
          f"| {'delta':>7}")
    hj = [r["vlm"] for r in recs]
    for f in FIELDS:
        h, t = dist(hj, f), dist(train, f)
        if not len(h) or not len(t):
            continue
        print(f"{f:<20} {h.mean():>14.2f} {h.std():>6.2f} | {t.mean():>11.2f} {t.std():>6.2f} "
              f"| {h.mean()-t.mean():>+7.2f}")
        out[f] = {"heldout_mean": float(h.mean()), "heldout_sd": float(h.std()),
                  "train_mean": float(t.mean()), "train_sd": float(t.std())}
    for tier in sorted(tiers):
        hj_t = [r["vlm"] for r in recs if r["tier"] == tier]
        line = "  ".join(f"{f.split('_')[0]}={dist(hj_t,f).mean():.2f}" for f in FIELDS
                         if len(dist(hj_t, f)))
        print(f"  tier {tier} (n={len(hj_t)}): {line}")

    print("\n--- histogram: aesthetic_score ---")
    for v in range(1, 10):
        hc = sum(1 for d in hj if d.get("aesthetic_score") == v)
        tc = sum(1 for d in train if d.get("aesthetic_score") == v)
        if hc or tc:
            print(f"  {v}: held-out {100*hc/len(hj):5.1f}%   training {100*tc/len(train):5.1f}%")

    print("\n--- categories (top 12) ---")
    print("  " + ", ".join(f"{k}:{v}" for k, v in
                           Counter(r["vlm"].get("category") for r in recs).most_common(12)))
    print("--- style ---")
    print("  " + ", ".join(f"{k}:{v}" for k, v in
                           Counter(r["vlm"].get("style") for r in recs).most_common()))

    print("\n=== MODALITY COMPLETENESS ===")
    nv = Counter(r["n_views"] for r in recs)
    print(f"n_views: {dict(nv)}")
    real_views = Counter(sum(1 for f in os.listdir(r["renders_dir"]) if f.endswith(".webp"))
                         for r in recs)
    print(f"renders actually on disk: {dict(real_views)}")
    ncap = Counter(len([c for c in caps.get(r["sha256"], []) if c]) for r in recs)
    print(f"captions per asset: {dict(ncap)}")
    uniq4 = sum(1 for r in recs if len(set(caps.get(r["sha256"], []))) == 4)
    print(f"assets whose 4 captions are all distinct (t003 has a texture caption): {uniq4}")

    def has(root, sha, k):
        return os.path.exists(os.path.join(root, sha[:2], sha, k))
    cov = {}
    for k, root in [("v000", CR), ("t000", CR), ("t001", CR), ("t002", CR), ("t003", CR),
                    ("m00", IMR)]:
        cov[k] = sum(1 for r in recs if has(root, r["sha256"], k + ".npz"))
        print(f"  {k:<5} present: {cov[k]}/{len(recs)}")
    nod = 0
    for r in recs:
        p = os.path.join(CR, r["sha256"][:2], r["sha256"], "v000.npz")
        if os.path.exists(p) and "dino_hidden" not in np.load(p).files:
            nod += 1
    print(f"  v000 missing the merged DINO segment: {nod}")
    out["coverage"] = cov
    print("\n=== GT LATENTS / PATHS (stat'd, not trusted) ===")
    bad = Counter()
    for r in recs:
        for k in ("ss_latent_64", "shape_latent_512", "pbr_latent_512"):
            if not os.path.isfile(r[k]):
                bad[k] += 1
        if not os.path.isdir(r["renders_dir"]):
            bad["renders_dir"] += 1
        if "/fsx/sfr/" in json.dumps(r):
            bad["dead_sfr_path"] += 1
    print(f"  missing/dead: {dict(bad) or 'none — all paths verified'}")

    print("\n=== DEDUP MARGIN OF THE SHIPPED SET ===")
    rc = np.array([r["dedup"]["render_cos"] for r in recs])
    gc = np.array([r["dedup"]["geom_cos_exact"] for r in recs])
    print(f"  render_cos to nearest training asset : max={rc.max():.3f} "
          f"p99={np.percentile(rc,99):.3f} median={np.median(rc):.3f}")
    print(f"  geom_cos(exact) to nearest training  : max={gc.max():.3f} "
          f"p99={np.percentile(gc,99):.3f} median={np.median(gc):.3f}")
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
