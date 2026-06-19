"""Precompute the frozen-DINOv3 token cache for a manifest (fusion cond, low-level segment).

Entries go into the SAME cache root as the VLM hiddens (same {sha[:2]}/{sha}/ dirs),
key d{view:03d}. Uses the EXACT live pieces — ImageTo3DDataset._load_views (crop/min-px)
and DinoV3FeatureExtractor (the extractor the pretrained TRELLIS cross-attn was trained
on, full token set incl. CLS/register) — so cached tokens are fp16-faithful to live.

  python scripts/build_dino_cache.py \
    --manifest .../ready_v1.jsonl --out_root data/vlm_hidden_cache/qwen35-2b_tok1024_crop \
    --crop_to_object 1 --shard 0 --num_shards 8
"""
from __future__ import annotations
import argparse, json, os, sys, time
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_THIS))
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import torch

from trellis2_blip3o import vlm_cache
from trellis2_blip3o.data.tasks.threed import ImageTo3DDataset
from trellis2_blip3o.dino_align import (
    DinoV3FeatureExtractor, TRELLIS_DINOV3_NAME, DINOV3_IMAGE_SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--crop_to_object", type=int, default=1)
    ap.add_argument("--max_views", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--min_aesthetic", type=float, default=None)
    args = ap.parse_args()
    dev = "cuda"

    ds = ImageTo3DDataset(manifest=args.manifest, ss_only=True,
                          crop_to_object=bool(args.crop_to_object),
                          min_aesthetic=args.min_aesthetic)
    ext = DinoV3FeatureExtractor(TRELLIS_DINOV3_NAME, image_size=DINOV3_IMAGE_SIZE)
    ext.model.eval().to(dev)
    for p in ext.model.parameters():
        p.requires_grad_(False)

    if args.shard == 0:
        # extend the EXISTING root meta with the dino contract (keep VLM fields intact)
        mp = os.path.join(args.out_root, "_meta.json")
        meta = json.load(open(mp)) if os.path.isfile(mp) else {}
        meta.update({"dino_model": TRELLIS_DINOV3_NAME,
                     "dino_image_size": DINOV3_IMAGE_SIZE,
                     "dino_max_views": int(args.max_views)})
        os.makedirs(args.out_root, exist_ok=True)
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)

    n_done = n_skip = 0
    t0 = time.time()
    for ri in range(args.shard, len(ds.records), args.num_shards):
        rec = ds.records[ri]
        sha = rec.get("sha256", f"idx_{ri}")
        n_views = min(int(rec.get("n_views", 16)), args.max_views)
        for v in range(n_views):
            key = vlm_cache.dino_key(v)
            if os.path.exists(vlm_cache.entry_path(args.out_root, sha, key)):
                n_skip += 1
                continue
            imgs = ds._load_views(rec["renders_dir"], [v])
            if not imgs:
                continue
            with torch.no_grad():
                feats = ext(imgs)                       # (1, N, 1024) fp32, full token set
            tok = feats[0]                              # (N, 1024)
            vlm_cache.save_entry(args.out_root, sha, key,
                                 tok, torch.ones(tok.shape[0], dtype=torch.bool))
            n_done += 1
            if n_done % 500 == 0:
                rate = n_done / max(1e-9, time.time() - t0)
                print(f"[shard {args.shard}] {n_done} done ({n_skip} skipped) "
                      f"{rate:.1f}/s", flush=True)
    print(f"[shard {args.shard}] DONE: {n_done} new, {n_skip} skipped, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
