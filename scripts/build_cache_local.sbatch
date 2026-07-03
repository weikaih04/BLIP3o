#!/bin/bash
#SBATCH --job-name=blip3o_cache
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=runs/cache_logs/slurm-%j.out
# ─────────────────────────────────────────────────────────────────────────────
# Build the OFFLINE conditioning cache on fsx (for stages ②③, and a faster Stage-1).
# Mirrors the GT Beaker jobs build_{vlm,dino}_cache_readyv1.yaml: 8 shards, one per GPU.
#   Phase V — build_vlm_cache.py  (v-keys: Qwen3.5-2B hidden per view)   [GPU]
#   Phase D — build_dino_cache.py (d-keys: DINOv3 tokens per view)        [GPU]
#   Phase M — merge_vd_cache.py   (merge v+d → one npz/view, IO fix)      [CPU]
#
# max_views=1 (weikaih 2026-06-26): ~2.6 TB total (v ~1.7 + d ~0.84) → fits /fsx/sfr (7.7T free).
# NO view augmentation (only view 0 cached). bit-faithful: ATTN_BACKEND=sdpa.
#
# Usage:  sbatch scripts/build_cache_local.sbatch
#   env overrides: MAX_VIEWS(1) NUM_SHARDS(8) MANIFEST OUT_ROOT MIN_AES(4.5)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o

export ATTN_BACKEND=sdpa                 # bit-faithful to live training (build_vlm_cache contract)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_VIEWS="${MAX_VIEWS:-1}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MANIFEST="${MANIFEST:-/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v2/ready_v2.jsonl}"
OUT_ROOT="${OUT_ROOT:-/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/qwen35-2b_tok1024_crop_mv1}"
MIN_AES="${MIN_AES:-4.5}"
LOGD=runs/cache_logs
mkdir -p "$OUT_ROOT" "$LOGD"
echo "[cache] node=$(hostname) max_views=$MAX_VIEWS shards=$NUM_SHARDS out=$OUT_ROOT"

echo "===== Phase V: VLM-hidden cache (v-keys) ====="
for i in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/build_vlm_cache.py \
    --manifest "$MANIFEST" --out_root "$OUT_ROOT" \
    --target_tokens_per_view 1024 --crop_to_object 1 --max_views "$MAX_VIEWS" \
    --min_aesthetic "$MIN_AES" --mode views \
    --shard "$i" --num_shards "$NUM_SHARDS" > "$LOGD/v_shard_$i.log" 2>&1 &
done
wait
echo "[cache] Phase V done"; tail -n 2 "$LOGD"/v_shard_*.log

echo "===== Phase D: DINOv3 cache (d-keys) ====="
for i in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/build_dino_cache.py \
    --manifest "$MANIFEST" --out_root "$OUT_ROOT" \
    --crop_to_object 1 --max_views "$MAX_VIEWS" \
    --min_aesthetic "$MIN_AES" \
    --shard "$i" --num_shards "$NUM_SHARDS" > "$LOGD/d_shard_$i.log" 2>&1 &
done
wait
echo "[cache] Phase D done"; tail -n 2 "$LOGD"/d_shard_*.log

echo "===== Phase M: merge v+d → one npz/view (CPU, 16 shards) ====="
for i in $(seq 0 15); do
  python scripts/merge_vd_cache.py \
    --root "$OUT_ROOT" --manifest "$MANIFEST" --max_views "$MAX_VIEWS" \
    --shard "$i" --num_shards 16 > "$LOGD/m_shard_$i.log" 2>&1 &
done
wait
echo "[cache] Phase M done — ALL CACHE BUILT at $OUT_ROOT"
du -sh "$OUT_ROOT" 2>/dev/null
cat "$OUT_ROOT/_meta.json" 2>/dev/null
