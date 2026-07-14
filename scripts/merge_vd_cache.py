"""Merge v-key (VLM hidden) + d-key (DINO tokens) cache entries into ONE npz per view.

Why: fusion training opens 2 small npz per sample; at 8-rank × 4-worker scale the
small-file random-read contention on weka cost +51% step time (measured 2026-06-11).
After merging, load_entry returns both segments from one open.

In-place and idempotent: rewrites v{view}.npz with the extra dino_* arrays (atomic),
skips entries already merged, leaves d-files in place (cheap; deleting is optional
via --delete_d after a verified pass). Pure IO — no GPU. Shardable.

  python scripts/merge_vd_cache.py --root data/vlm_hidden_cache/qwen35-2b_tok1024_crop \
      --manifest .../ready_v1.jsonl --max_views 4 --shard 0 --num_shards 16
"""
from __future__ import annotations
import argparse, json, os, sys, time
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_THIS))

import numpy as np

from trellis2_blip3o import vlm_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--d_root", default=None,
                    help="root holding the d-files (default: --root). Lets a NEW v-cache "
                         "(e.g. rebuilt with a different VLM) reuse the VLM-independent DINO "
                         "entries of an existing cache without copying them.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max_views", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--delete_d", action="store_true",
                    help="remove the d-file after verifying the merged v-file")
    ap.add_argument("--force", action="store_true",
                    help="re-merge even when dino_hidden is already present (overwrite the "
                         "dino segment from d_root — for replacing a stale DINO merge)")
    args = ap.parse_args()
    d_root = args.d_root or args.root
    # STALE-DINO GUARD (2026-07-13): renders_cond was re-rendered ~07-12; the d-files in these
    # roots were extracted from the OLD renders and must never be merged again. Use the
    # freshly rebuilt dino_fix_* roots instead. Exit 0 so batch pipelines proceed as no-ops.
    for stale in ("v22_3dvlm_tok1024_mv1", "v22_heldout"):
        if stale in d_root:
            print(f"[merge] REFUSING stale DINO d_root '{d_root}' (renders re-rendered 07-12; "
                  f"use dino_fix_*). No-op exit.", flush=True)
            return

    recs = [json.loads(l) for l in open(args.manifest)]
    n_done = n_skip = n_missing_d = 0
    t0 = time.time()
    for ri in range(args.shard, len(recs), args.num_shards):
        rec = recs[ri]
        sha = rec.get("sha256")
        nv = min(int(rec.get("n_views", 16)), args.max_views)
        for v in range(nv):
            vp = vlm_cache.entry_path(args.root, sha, vlm_cache.view_key(v))
            dp = vlm_cache.entry_path(d_root, sha, vlm_cache.dino_key(v))
            try:
                va = np.load(vp)
            except FileNotFoundError:
                continue
            if "dino_hidden" in va.files and not args.force:
                n_skip += 1
                if args.delete_d and os.path.exists(dp):
                    os.remove(dp)
                continue
            try:
                da = np.load(dp)
            except FileNotFoundError:
                n_missing_d += 1
                continue
            tmp = vp + ".tmp"
            np.savez(tmp, hidden=va["hidden"], keep_mask=va["keep_mask"],
                     dino_hidden=da["hidden"], dino_keep_mask=da["keep_mask"])
            os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, vp)
            if args.delete_d:
                os.remove(dp)
            n_done += 1
            if n_done % 2000 == 0:
                print(f"[shard {args.shard}] merged {n_done} (skip {n_skip}) "
                      f"{n_done/max(1e-9, time.time()-t0):.0f}/s", flush=True)
    print(f"[shard {args.shard}] DONE: merged {n_done}, already {n_skip}, "
          f"missing-d {n_missing_d}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
