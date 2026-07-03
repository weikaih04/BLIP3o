#!/bin/bash
# Oversubscribed cond-cache build for ready_v2_clean.jsonl. The build is DATA-loading-bound
# (GPU ~3% util), so NSHARD shards (shard i pinned to GPU i%8) parallelize the /fsx render
# read + CPU preprocess. V (Qwen) -> D (DINO) -> M (merge). All phases resumable (skip existing).
set -uo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export ATTN_BACKEND=sdpa TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=3
MANIFEST=/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v2/ready_v2_clean.jsonl
OUT_ROOT=/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/qwen35-2b_tok1024_crop_mv1
NSHARD="${NSHARD:-16}"
LOGD=runs/cache_logs/v2fast
mkdir -p "$LOGD"
echo "[fast] NSHARD=$NSHARD (GPU=i%8) manifest=$(basename "$MANIFEST")"

echo "===== Phase V (Qwen) ====="
for i in $(seq 0 $((NSHARD-1))); do
  CUDA_VISIBLE_DEVICES=$((i % 8)) python scripts/build_vlm_cache.py \
    --manifest "$MANIFEST" --out_root "$OUT_ROOT" \
    --target_tokens_per_view 1024 --crop_to_object 1 --max_views 1 \
    --min_aesthetic 4.5 --mode views --shard "$i" --num_shards "$NSHARD" \
    > "$LOGD/v_$i.log" 2>&1 &
done
wait
echo "[fast] Phase V done"

echo "===== Phase D (DINO) ====="
for i in $(seq 0 $((NSHARD-1))); do
  CUDA_VISIBLE_DEVICES=$((i % 8)) python scripts/build_dino_cache.py \
    --manifest "$MANIFEST" --out_root "$OUT_ROOT" \
    --crop_to_object 1 --max_views 1 --min_aesthetic 4.5 \
    --shard "$i" --num_shards "$NSHARD" > "$LOGD/d_$i.log" 2>&1 &
done
wait
echo "[fast] Phase D done"

echo "===== Phase M (merge, CPU 32 shards) ====="
for i in $(seq 0 31); do
  python scripts/merge_vd_cache.py --root "$OUT_ROOT" --manifest "$MANIFEST" \
    --max_views 1 --shard "$i" --num_shards 32 > "$LOGD/m_$i.log" 2>&1 &
done
wait
echo "[fast] ALL CACHE BUILT"
du -sh "$OUT_ROOT" 2>/dev/null
