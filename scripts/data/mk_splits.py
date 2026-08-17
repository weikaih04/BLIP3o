#!/usr/bin/env python
"""Carve val/test out of the 400k tier and strip them from ALL THREE pools.

WHY THIS EXISTS (2026-08-17). The 1% held-out we had was cut only from the 400k
tier; `pretrain_clean` (1.8M) and `vlm_filtered_all_matclean` (800k) still
contained 99.9% of it. That is harmless only as long as nobody trains on the
upstream pools — and the mix config already plans to ("when the SWH cache
exists, regenerate and the pool doubles"). The first pretrain run or pool
expansion would have silently turned the eval set into training data.

THE THREE POOLS ARE EXACTLY NESTED (measured, 0 exceptions):
    1.8M pretrain_clean            1,667,854
     |__ 800k all_matclean           787,472
          |__ 400k (has cond cache)  397,238
so defining the split on the SMALLEST tier and removing it from all three is
sufficient and leak-free at every tier.

SPLIT DEFINED ON THE 400k TIER, not the top: only that tier has cond caches, so
only its assets are evaluable at all. Assets outside it can never be val/test.

    r = int(md5(sha)[:8], 16) % 1000
    val   r <  25          ~2.5%
    test  25 <= r < 50     ~2.5%
    train r >= 50

BACKWARD COMPAT — READ BEFORE COMPARING RUNS. The OLD held-out was
`md5(sha)[:8] % 100 == 0`, i.e. r in {0,100,...,900}. Under the new rule only
r==0 lands in val; the other 90% returns to train. So the 2026-08-16 60K
checkpoint (which trained on everything except the old 1%) can only be compared
against future runs on the r==0 slice (~400 assets) — enough for n=200, not more.
This is deliberate: a clean rule beats a grandfathered one, and the next run
changes the config anyway.

VAL/TEST EXCLUDE the reverse material bug (skill S3.1: cond_sat>0.12 and
gt_sat<0.06, 8,101 pool-wide, ALL labelled `fine`). Those assets' GT is wrong,
so a metric computed on them PENALISES the correct answer — measured: on the
gold lamp, the model that faithfully reproduced the golden input scored 1.44
while the one that output grey scored 0.90.

Usage (compute node — this streams 1.8M rows):
  srun --jobid=<hold> --overlap python scripts/mk_splits.py
"""
import hashlib
import json
import os
import sys

OUT = "manifests/splits"
UP = "/fsx/home/weikai.huang/3dgen/data/trellis2/manifests"
POOL18 = f"{UP}/ready_v5_pretrain/pretrain_clean.jsonl"
POOL8 = f"{UP}/ready_v5_vlm_filtered/vlm_filtered_all_matclean.jsonl"
SRC400 = ["manifests/v5_matclean_cached_train.jsonl",
          "manifests/v5_matclean_cached_heldout.jsonl"]
# (the v4 capT source that used to live here is gone — see the capT note in
#  main(); the text lists come from the caption store now)
VERDICT = "/fsx/home/weikai.huang/3dgen/data/_decgt/material_verdict_v5.jsonl"

VAL_HI, TEST_HI = 25, 50          # per-mille cuts


def bucket(sha: str) -> int:
    return int(hashlib.md5(sha.encode()).hexdigest()[:8], 16) % 1000


def rows(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                yield json.loads(line)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- reverse-material-bug set (both fields can be null; comparing a None
    # to a float raises TypeError — skill S3.1 records that exact crash) -------
    bad = set()
    with open(VERDICT) as f:
        for line in f:
            d = json.loads(line)
            cs, gs = d.get("cond_sat"), d.get("gt_sat")
            if cs is None or gs is None:
                continue
            if cs > 0.12 and gs < 0.06:
                bad.add(d["sha"])
    print(f"[splits] reverse-material-bug assets pool-wide: {len(bad)}", flush=True)

    # ---- 400k tier: define the split ---------------------------------------
    val, test, tr400 = [], [], []
    dropped = 0
    seen = set()
    for r in rows(SRC400):
        sha = r["sha256"]
        b = bucket(sha)
        if b < TEST_HI:                  # i.e. it would land in val or test
            if sha in bad:
                dropped += 1
                tr400.append(r)          # keep it OUT of eval, not out of train
                continue
        (val if b < VAL_HI else test if b < TEST_HI else tr400).append(r)
        seen.add(sha)
    holdout = {r["sha256"] for r in val} | {r["sha256"] for r in test}
    print(f"[splits] 400k tier: val {len(val)}  test {len(test)}  train {len(tr400)}"
          f"   (dropped {dropped} broken-GT rows from eval)", flush=True)

    for name, data in (("val", val), ("test", test), ("pool400k_train", tr400)):
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

    # ---- text task: NOT built here anymore ---------------------------------
    # This used to slice ready_v4's capT manifest by the same sha rule, which is
    # what pinned the text task to the v4 asset set. Captions do not live in any
    # manifest — they are a standalone store covering 1,741,113 assets — so the
    # text lists are now re-joined onto THESE pools by
    # scripts/mk_capT_from_store.py. Run that after this script.
    print("[splits] capT: skipped — run scripts/mk_capT_from_store.py "
          "(joins the caption store onto these pools; no v4 involved)", flush=True)

    # ---- upstream pools: strip the SAME sha set ----------------------------
    # streamed, never held in memory: pretrain_clean is 1.77 GB
    for tag, src in (("pool800k_train", POOL8), ("pool1800k_train", POOL18)):
        n = kept = 0
        with open(src) as fi, open(f"{OUT}/{tag}.jsonl", "w") as fo:
            for line in fi:
                n += 1
                sha = json.loads(line).get("sha256")
                if sha in holdout:
                    continue
                kept += 1
                fo.write(line)
        print(f"[splits] {tag}: {n} -> {kept}  (removed {n - kept})", flush=True)

    # ---- fixed stratified subsets ------------------------------------------
    # proportional by subset with a floor of 2, so the rare subsets (Toys4k 13,
    # SWH 4) are not silently absent from a 200-draw. Deterministic order =
    # hash, so the same sha list comes back on every rebuild.
    def stratify(pool, k):
        # DEDUPE BY SHA FIRST. The source manifest has multi-row assets
        # (394,902 rows / 393,284 unique sha), so a naive draw returns the same
        # asset twice — silently reweighting the average and shrinking the
        # effective n. Measured before this fix: val200 held 199 unique sha,
        # val512 held 508.
        seen_sha, uniq = set(), []
        for r in pool:
            if r["sha256"] in seen_sha:
                continue
            seen_sha.add(r["sha256"])
            uniq.append(r)
        pool = uniq
        by = {}
        for r in pool:
            by.setdefault(r.get("subset", "?"), []).append(r)
        for v in by.values():
            v.sort(key=lambda r: hashlib.md5(("s" + r["sha256"]).encode()).hexdigest())
        alloc = {s: min(2, len(v)) for s, v in by.items()}
        rest = k - sum(alloc.values())
        tot = sum(max(0, len(v) - alloc[s]) for s, v in by.items())
        for s, v in by.items():
            if rest <= 0 or tot <= 0:
                break
            add = min(len(v) - alloc[s], round(rest * (len(v) - alloc[s]) / tot))
            alloc[s] += add
        out = []
        for s, v in by.items():
            out.extend(v[:alloc[s]])
        # rounding leaves a shortfall (measured: test512 came out 511); top up
        # deterministically from whatever strata still have rows
        if len(out) < k:
            taken = {r["sha256"] for r in out}
            for s, v in sorted(by.items(), key=lambda x: -len(x[1])):
                for r in v:
                    if len(out) >= k:
                        break
                    if r["sha256"] not in taken:
                        taken.add(r["sha256"])
                        out.append(r)
        return out[:k]

    for base, pool in (("val", val), ("test", test)):
        for k in (200, 512):
            sub = stratify(pool, k)
            with open(f"{OUT}/{base}{k}.jsonl", "w") as f:
                for r in sub:
                    f.write(json.dumps(r) + "\n")
            dist = {}
            for r in sub:
                dist[r.get("subset")] = dist.get(r.get("subset"), 0) + 1
            print(f"[splits] {base}{k}: {len(sub)}  {dict(sorted(dist.items(), key=lambda x: -x[1]))}",
                  flush=True)

    print(f"\n[splits] wrote -> {OUT}/")


if __name__ == "__main__":
    main()
