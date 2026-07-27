"""Reduce sharded diag_im_multiview / eval_im_vs_i1_ss runs and MEASURE THE NOISE FLOOR.

The point of the clean held-out set is a statistic we can trust.  This script answers,
empirically: what effect size can it resolve?

Inputs: several runs of the SAME probe over the SAME manifest that differ only in things
that should NOT change the answer — checkpoint (two nearby converged ckpts of one run) and
sampler seed.  Any spread across those runs is noise.

  python scripts/reduce_diag_im.py --glob 'runs/cache_logs/diag_im_mv_checkpoint-*.json' \
      --group_by ckpt,seed --manifest .../heldout_clean.jsonl

Reports, for the 4d-4c statistic and for the raw IoUs:
  * per-run mean, and the SPREAD (range / sd) across runs        -> the noise floor
  * sd of the per-asset paired difference / sqrt(n)              -> the sampling-error
                                                                    floor the set size buys
  * per-asset sd across runs, sorted                             -> unstable assets
  * the same numbers restricted to any subgroup (tier, subset)
"""
import argparse
import glob as _glob
import json
import os
from collections import defaultdict

import numpy as np


def load_runs(pattern):
    runs = []
    for p in sorted(_glob.glob(pattern)):
        d = json.load(open(p))
        if "per_asset" not in d:
            continue
        d["_path"] = p
        runs.append(d)
    return runs


def merge_shards(runs):
    """Runs that differ only by `shard` are one logical run -> concatenate per_asset."""
    by = defaultdict(list)
    for r in runs:
        key = (r.get("im_ckpt") or r.get("probe_ckpt"), r.get("seed", 0))
        by[key].append(r)
    out = []
    for (ck, seed), rs in sorted(by.items()):
        pa = {}
        for r in rs:
            for a in r["per_asset"]:
                pa[a["sha8"]] = a
        out.append({"ckpt": ck, "seed": seed, "n_shards": len(rs), "per_asset": pa,
                    "paths": [r["_path"] for r in rs]})
    return out


def stat_block(name, vals):
    v = np.asarray(vals, dtype=np.float64)
    return (f"{name:<26} n={len(v):<4} mean={v.mean():+.4f} sd={v.std(ddof=1):.4f} "
            f"min={v.min():+.4f} max={v.max():+.4f} range={v.max()-v.min():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--manifest", default=None,
                    help="held-out manifest, to break the numbers down by tier/subset")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = merge_shards(load_runs(a.glob))
    if not runs:
        raise SystemExit(f"no runs matched {a.glob}")
    common = set(runs[0]["per_asset"])
    for r in runs[1:]:
        common &= set(r["per_asset"])
    common = sorted(common)
    print(f"[reduce] {len(runs)} logical runs, {len(common)} assets common to all\n")
    for r in runs:
        print(f"  run ckpt={os.path.basename(str(r['ckpt']).rstrip('/'))} seed={r['seed']} "
              f"shards={r['n_shards']} assets={len(r['per_asset'])}")
    print()

    meta = {}
    if a.manifest:
        for line in open(a.manifest):
            rec = json.loads(line)
            meta[rec["sha256"][:8]] = rec

    # which keys exist: diag (4distinct/4copy/1view) or eval (I1_baseline/probe_IM_4view)
    k0 = runs[0]["per_asset"][common[0]]
    is_diag = "4distinct" in k0
    keys = (["4distinct", "4copy", "1view"] if is_diag
            else ["I1_baseline", "probe_I1", "probe_IM_4view"])
    dname, dpair = (("4d-4c", ("4distinct", "4copy")) if is_diag
                    else ("IM4v-I1base", ("probe_IM_4view", "I1_baseline")))

    # (runs, assets) matrices
    M = {k: np.array([[r["per_asset"][s][k] for s in common] for r in runs]) for k in keys}
    D = M[dpair[0]] - M[dpair[1]]

    print("=== PER-RUN MEANS (each row is a run that SHOULD give the same answer) ===")
    for i, r in enumerate(runs):
        row = "  ".join(f"{k}={M[k][i].mean():.4f}" for k in keys)
        print(f"  ckpt={os.path.basename(str(r['ckpt']).rstrip('/')):<18} seed={r['seed']}  "
              f"{row}  {dname}={D[i].mean():+.4f}")
    print()

    print("=== NOISE FLOOR (spread of the run-level mean across runs) ===")
    print("  " + stat_block(f"mean {dname}", D.mean(1)))
    for k in keys:
        print("  " + stat_block(f"mean {k}", M[k].mean(1)))
    print()

    print("=== SAMPLING-ERROR FLOOR (what n buys, per run) ===")
    for i, r in enumerate(runs):
        se = D[i].std(ddof=1) / np.sqrt(len(common))
        print(f"  ckpt={os.path.basename(str(r['ckpt']).rstrip('/')):<18} seed={r['seed']}  "
              f"sd_per_asset({dname})={D[i].std(ddof=1):.4f}  SE(mean)={se:.4f}  "
              f"95%CI=+-{1.96*se:.4f}")
    print()

    print("=== PER-ASSET STABILITY (sd across runs; the worst are the unstable assets) ===")
    per = np.stack([D[:, j] for j in range(len(common))])     # (assets, runs)
    sd = per.std(1, ddof=1) if len(runs) > 1 else np.zeros(len(common))
    ious = np.stack([M[keys[0]][:, j] for j in range(len(common))])
    iousd = ious.std(1, ddof=1) if len(runs) > 1 else np.zeros(len(common))
    order = np.argsort(-sd)
    print(f"  median per-asset sd of {dname} across runs = {np.median(sd):.4f}; "
          f"of {keys[0]} = {np.median(iousd):.4f}")
    print(f"  {'sha8':<10} {'sd(' + dname + ')':>12} {'sd(' + keys[0] + ')':>14} "
          f"{'mean ' + keys[0]:>14}  tier subset")
    for j in order[:15]:
        m = meta.get(common[j], {})
        print(f"  {common[j]:<10} {sd[j]:>12.4f} {iousd[j]:>14.4f} "
              f"{ious[j].mean():>14.4f}  {m.get('tier','?')}    {m.get('subset','?')}")
    print()

    # excluding the k most unstable assets, how much does the noise floor drop?
    print("=== NOISE FLOOR AFTER DROPPING THE k MOST UNSTABLE ASSETS ===")
    for k in (0, 1, 2, 5, 10, 20):
        if k >= len(common):
            break
        keep = order[k:]
        dm = D[:, keep].mean(1)
        print(f"  drop {k:>2}: n={len(keep):<4} mean {dname} range={dm.max()-dm.min():.4f} "
              f"sd={dm.std(ddof=1) if len(runs)>1 else 0:.4f}  "
              f"SE(mean)={D[:, keep].std(1, ddof=1).mean()/np.sqrt(len(keep)):.4f}")
    print()

    if meta:
        print("=== BY SUBGROUP (first run) ===")
        for field in ("tier", "subset"):
            groups = defaultdict(list)
            for j, s in enumerate(common):
                groups[meta.get(s, {}).get(field, "?")].append(j)
            for g, idx in sorted(groups.items()):
                dm = D[:, idx].mean(1)
                print(f"  {field}={g:<22} n={len(idx):<4} mean {dname}={dm.mean():+.4f} "
                      f"run-range={dm.max()-dm.min():.4f}  "
                      f"mean {keys[0]}={M[keys[0]][:, idx].mean():.4f}")
        print()

    if a.out:
        json.dump({"n": len(common), "runs": [{"ckpt": r["ckpt"], "seed": r["seed"]}
                                              for r in runs],
                   "delta_name": dname,
                   "per_run_mean_delta": D.mean(1).tolist(),
                   "noise_floor_range": float(D.mean(1).max() - D.mean(1).min()),
                   "noise_floor_sd": float(D.mean(1).std(ddof=1)) if len(runs) > 1 else 0.0,
                   "se_mean_per_run": (D.std(1, ddof=1) / np.sqrt(len(common))).tolist(),
                   "per_asset": {s: {"sd_delta": float(sd[j]),
                                     "sd_" + keys[0]: float(iousd[j]),
                                     "mean_" + keys[0]: float(ious[j].mean())}
                                 for j, s in enumerate(common)}},
                  open(a.out, "w"), indent=1)
        print(f"[reduce] wrote {a.out}")


if __name__ == "__main__":
    main()
