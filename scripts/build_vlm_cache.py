"""Precompute the VLM-hidden cache for a manifest (I1 single-view entries).

Reuses the EXACT live pipeline pieces — ImageTo3DDataset._load_views (crop/min-px),
collate_vlm_3d (template/budget/upscale), model.encode_cond (VLM + mask) — so cached
hiddens are bit-faithful (fp16-rounded) to what live training would compute.

Shardable: run N copies with --shard i --num_shards N (one GPU each).

  python scripts/build_vlm_cache.py \
    --manifest .../ready_v1.jsonl --out_root data/vlm_hidden_cache/qwen35-2b_tok1024_crop \
    --target_tokens_per_view 1024 --crop_to_object 1 --shard 0 --num_shards 8
"""
from __future__ import annotations
import argparse, json, os, sys, time
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_THIS))
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import torch
from transformers import AutoProcessor

from trellis2_blip3o import vlm_cache
from trellis2_blip3o.vlm_collate import collate_vlm_3d
from trellis2_blip3o.data.tasks.threed import ImageTo3DDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--vlm", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--target_tokens_per_view", type=int, default=1024)
    ap.add_argument("--crop_to_object", type=int, default=1)
    ap.add_argument("--max_views", type=int, default=8)   # cache first N views per asset
    ap.add_argument("--mode", default="views", choices=["views", "captions", "combos"],
                    help="views = I1 per-view hiddens (v-keys); captions = text-only "
                         "hiddens per caption-list index (t-keys, for the T task); "
                         "combos = IM pinned multi-view combos (m-keys: m00=2 views, "
                         "m01=3, m02=4; sha-seeded views; joint VLM hidden + concat "
                         "per-view DINO from the merged v-entries — ONE file per sample)")
    ap.add_argument("--im_token_budget", type=int, default=2048,
                    help="combos mode: TOTAL Qwen vision-token budget split over views")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--min_aesthetic", type=float, default=None)
    args = ap.parse_args()
    dev = "cuda"

    # dataset object ONLY for its record list + _load_views (live crop pipeline)
    ds = ImageTo3DDataset(manifest=args.manifest, ss_only=True,
                          crop_to_object=bool(args.crop_to_object),
                          min_aesthetic=args.min_aesthetic)
    proc = AutoProcessor.from_pretrained(args.vlm)

    # VLM only (no flows): TrellisNativeVLM would build ss_flow too — load the bare
    # VLM and replicate encode_cond? NO — guaranteed-identity matters more than 2.6GB:
    # build the composite with build_slat=False and use model.encode_cond verbatim.
    from blip3o.model.language_model.trellis_native_vlm import (
        TrellisNativeVLMForConditionalGeneration, TrellisNativeVLMConfig)
    cfg = TrellisNativeVLMConfig(vlm_model=args.vlm, freeze_vlm=True, build_slat=False,
                                 target_tokens_per_view=args.target_tokens_per_view)
    model = TrellisNativeVLMForConditionalGeneration(cfg).to(dev).eval()

    # combos mode: budgeted per-view DINO sizes (multiples of 16; ≈1k tokens total)
    # 4K-total tier (weikaih 2026-06-12): Qwen budget 2048 + DINO ≈2k →
    # totals 2v≈4.1k / 3v≈4.1k / 4v≈4.2k (constant). Extra capacity goes to DINO
    # (the fidelity carrier per the segment ablations), not Qwen.
    IM_DINO_SIZES = {2: 512, 3: 416, 4: 368}
    dino_ext = None
    if args.mode == "combos":
        from trellis2_blip3o.dino_align import DinoV3FeatureExtractor, TRELLIS_DINOV3_NAME
        dino_ext = DinoV3FeatureExtractor(TRELLIS_DINOV3_NAME, image_size=512)
        dino_ext.model.eval().to(dev)
        for p_ in dino_ext.model.parameters():
            p_.requires_grad_(False)

    if args.shard == 0:
        if args.mode == "views":
            vlm_cache.write_meta(args.out_root, vlm=args.vlm,
                                 target_tokens_per_view=args.target_tokens_per_view,
                                 crop_to_object=bool(args.crop_to_object),
                                 max_views=args.max_views)
        else:
            # captions/combos extend the EXISTING meta — never clobber the views/dino contract
            mp = os.path.join(args.out_root, "_meta.json")
            meta = json.load(open(mp)) if os.path.isfile(mp) else {}
            meta.update({"has_captions": True} if args.mode == "captions"
                        else {"im_combo_sizes": [2, 3, 4],
                              "im_token_budget": int(args.im_token_budget),
                              "im_dino_sizes": {"2": 512, "3": 416, "4": 368}})
            os.makedirs(args.out_root, exist_ok=True)
            with open(mp, "w") as f:
                json.dump(meta, f, indent=2)

    n_done = n_skip = 0
    t0 = time.time()
    for ri in range(args.shard, len(ds.records), args.num_shards):
        rec = ds.records[ri]
        sha = rec.get("sha256", f"idx_{ri}")
        if args.mode == "captions":
            # T task: one entry per caption-list index (dataset samples by index).
            caps = [c for c in (rec.get("captions") or []) if c]
            for ci, cap in enumerate(caps):
                key = vlm_cache.caption_key(ci)
                if os.path.exists(vlm_cache.entry_path(args.out_root, sha, key)):
                    n_skip += 1
                    continue
                batch = collate_vlm_3d([{"images": [], "caption": cap}], proc,
                                       target_tokens_per_view=args.target_tokens_per_view)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden, _ = model.encode_cond(
                        input_ids=batch["input_ids"].to(dev),
                        attention_mask=batch["attention_mask"].to(dev),
                        pixel_values=None, image_grid_thw=None,
                    )
                T = int(batch["attention_mask"][0].sum())
                vlm_cache.save_entry(args.out_root, sha, key,
                                     hidden[0, :T], batch["cond_keep_mask"][0, :T])
                n_done += 1
                if n_done % 1000 == 0:
                    print(f"[shard {args.shard}] {n_done} done ({n_skip} skipped) "
                          f"{n_done/max(1e-9, time.time()-t0):.1f}/s", flush=True)
            continue
        if args.mode == "combos":
            import numpy as _np
            n_avail = min(int(rec.get("n_views", 16)), args.max_views)
            crng = _np.random.default_rng(
                int.from_bytes(__import__("hashlib").sha256(sha.encode()).digest()[:4], "little"))
            # sha-seeded via hashlib (stable across processes; built-in hash() is
            # PYTHONHASHSEED-randomized → combos would differ per rebuild)
            for ci, size in enumerate((2, 3, 4)):
                if size > n_avail:
                    continue
                key = vlm_cache.combo_key(ci)
                if os.path.exists(vlm_cache.entry_path(args.out_root, sha, key)):
                    n_skip += 1
                    continue
                views = sorted(int(v) for v in crng.choice(n_avail, size=size, replace=False))
                imgs = ds._load_views(rec["renders_dir"], views)
                if len(imgs) != size:
                    continue
                # BUDGETED DINO (weikaih 2026-06-12: IM cond must NOT grow linearly with
                # view count): per-view DINO at a size keeping the segment ≈ single-image
                # scale — total ≈ 1.0-1.1k tokens for any combo size.
                dsize = IM_DINO_SIZES[size]
                dino_ext.image_size = dsize       # extractor resizes PIL to this square
                with torch.no_grad():
                    dfeats = dino_ext(imgs)       # (V, N_s, 1024) fp32
                dino_h = dfeats.reshape(-1, dfeats.shape[-1])
                dino_m = torch.ones(dino_h.shape[0], dtype=torch.bool)
                dino_ids = torch.cat([torch.full((dfeats.shape[1],), oi, dtype=torch.long)
                                      for oi in range(size)], 0)
                # BUDGETED Qwen: token_budget 2048 split over views (collator handles:
                # per_tok = min(4096, 2048//n) → 2v:1024/v, 3v:682/v, 4v:512/v).
                batch = collate_vlm_3d([{"images": imgs, "caption": ""}], proc,
                                       target_tokens_per_view=args.target_tokens_per_view,
                                       token_budget=args.im_token_budget)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden, _ = model.encode_cond(
                        input_ids=batch["input_ids"].to(dev),
                        attention_mask=batch["attention_mask"].to(dev),
                        pixel_values=batch.get("pixel_values").to(dev) if batch.get("pixel_values") is not None else None,
                        image_grid_thw=batch.get("image_grid_thw").to(dev) if batch.get("image_grid_thw") is not None else None,
                    )
                T = int(batch["attention_mask"][0].sum())
                vlm_cache.save_entry(args.out_root, sha, key,
                                     hidden[0, :T], batch["cond_keep_mask"][0, :T],
                                     views=views, dino_hidden=dino_h,
                                     dino_keep_mask=dino_m, dino_view_ids=dino_ids)
                n_done += 1
                if n_done % 500 == 0:
                    print(f"[shard {args.shard}] {n_done} done ({n_skip} skipped) "
                          f"{n_done/max(1e-9, time.time()-t0):.1f}/s", flush=True)
            continue
        n_views = min(int(rec.get("n_views", 16)), args.max_views)
        for v in range(n_views):
            if os.path.exists(vlm_cache.entry_path(args.out_root, sha, v)):
                n_skip += 1
                continue
            imgs = ds._load_views(rec["renders_dir"], [v])
            if not imgs:
                continue
            batch = collate_vlm_3d([{"images": imgs, "caption": ""}], proc,
                                   target_tokens_per_view=args.target_tokens_per_view)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden, _ = model.encode_cond(
                    input_ids=batch["input_ids"].to(dev),
                    attention_mask=batch["attention_mask"].to(dev),
                    pixel_values=batch.get("pixel_values").to(dev) if batch.get("pixel_values") is not None else None,
                    image_grid_thw=batch.get("image_grid_thw").to(dev) if batch.get("image_grid_thw") is not None else None,
                )
            T = int(batch["attention_mask"][0].sum())          # unpadded true length
            vlm_cache.save_entry(args.out_root, sha, v,
                                 hidden[0, :T], batch["cond_keep_mask"][0, :T])
            n_done += 1
            if n_done % 200 == 0:
                rate = n_done / max(1e-9, time.time() - t0)
                print(f"[shard {args.shard}] {n_done} done ({n_skip} skipped) "
                      f"{rate:.1f}/s", flush=True)
    print(f"[shard {args.shard}] DONE: {n_done} new, {n_skip} skipped, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
