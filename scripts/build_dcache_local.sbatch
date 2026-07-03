#!/bin/bash
#SBATCH --job-name=blip3o_dcache
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=runs/cache_logs/slurm-dcache-%j.out
# ─────────────────────────────────────────────────────────────────────────────
# DINOv3 d-cache ONLY (Phase D), 8 shards. Run this on a SECOND node CONCURRENTLY
# with the v-cache (build_cache_local.sbatch Phase V) — they write disjoint keys
# (v* vs d*) into the same out_root, so total wall-clock = max(v,d) not v+d.
# Idempotent: when the v-cache job reaches its own Phase D it fast-skips these.
# Merge (Phase M) is done by build_cache_local.sbatch after both v+d exist.
#
# Usage: sbatch scripts/build_dcache_local.sbatch
#   env: MAX_VIEWS(1) NUM_SHARDS(8) MANIFEST OUT_ROOT MIN_AES(4.5)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export ATTN_BACKEND=sdpa
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_VIEWS="${MAX_VIEWS:-1}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MANIFEST="${MANIFEST:-/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v2/ready_v2.jsonl}"
OUT_ROOT="${OUT_ROOT:-/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/qwen35-2b_tok1024_crop_mv1}"
MIN_AES="${MIN_AES:-4.5}"
LOGD=runs/cache_logs
mkdir -p "$OUT_ROOT" "$LOGD"
echo "[dcache] node=$(hostname) max_views=$MAX_VIEWS shards=$NUM_SHARDS out=$OUT_ROOT"

echo "===== Phase D (standalone): DINOv3 cache (d-keys) ====="
for i in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/build_dino_cache.py \
    --manifest "$MANIFEST" --out_root "$OUT_ROOT" \
    --crop_to_object 1 --max_views "$MAX_VIEWS" \
    --min_aesthetic "$MIN_AES" \
    --shard "$i" --num_shards "$NUM_SHARDS" > "$LOGD/d_shard_$i.log" 2>&1 &
done
wait
echo "[dcache] Phase D done"; tail -n 2 "$LOGD"/d_shard_*.log
