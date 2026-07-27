# The clean held-out evaluation set

Replaces `im_probe/heldout14*.jsonl` (n=14), which was too small **and** contaminated.

* manifest: `/fsx/home/weikai.huang/3dgen/im_probe/heldout_clean/heldout_clean.jsonl`
* captions: `/fsx/home/weikai.huang/3dgen/im_probe/heldout_clean/heldout_clean_capT.jsonl`
* README with the criterion, thresholds and measured numbers:
  `/fsx/home/weikai.huang/3dgen/im_probe/heldout_clean/README.md`

Build (reproducible from scratch):

```bash
python scripts/build_heldout_clean.py --stage pool
python scripts/build_heldout_clean.py --stage feats --which train   # ~422k, ~28 min, 32 procs
python scripts/build_heldout_clean.py --stage feats --which cand
python scripts/build_heldout_clean.py --stage dedup
python scripts/build_heldout_clean.py --stage manifest
GPUS=0,1,2,3 bash scripts/build_heldout_clean_cache.sh     # I1 + IM + T cond caches
python scripts/heldout_clean_report.py                     # characterisation
```

Measure what the set can resolve:

```bash
GPUS=0,1,2,3 bash scripts/heldout_noise_floor.sh           # 2 ckpts x 2 seeds, then reduce
python scripts/reduce_diag_im.py --glob 'runs/cache_logs/diag_im_mv_clean_*.json' \
    --manifest /fsx/home/weikai.huang/3dgen/im_probe/heldout_clean/heldout_clean.jsonl
```

Eyeball the dedup decisions (do this whenever a threshold changes):

```bash
python scripts/dedup_spotcheck.py --band geom_only   --n 8 --out /tmp/geom.png
python scripts/dedup_spotcheck.py --band render_only --n 8 --out /tmp/render.png
python scripts/dedup_spotcheck.py --band nearmiss    --n 8 --out /tmp/near.png
```

## Using it

```bash
MANI=/fsx/home/weikai.huang/3dgen/im_probe/heldout_clean/heldout_clean.jsonl
COND_I1=/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_heldout
COND_IM=/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_clean_im4r    # WEIGHTED views
```

`COND_IM` matters: the old `v22_heldout_im4l` used the legacy fixed good-band view pick
while training (`v22_im4r`) used weighted 4-of-16. The new cache matches training
(`im_view_sampling: weighted`, `im_qwen_tok_per_view: 64`, `dino_image_size: 320`).

Every record carries `tier` (`A` = drawn under the exact training quality gate,
`B` = relaxed top-up with a documented quality skew) and `dedup` (its similarity to the
nearest training asset on each axis). Report tier-A-only numbers when the quality skew
could matter.
