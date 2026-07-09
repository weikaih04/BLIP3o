#!/bin/bash
# v2.2 cond cache for ready_v4_vlm_filtered (422k). Per-node: 8 GPU-shards do V→D→M on
# their own assets (consistent sharding → per-shard merge is independent, /fsx shared).
# Launch across 2 nodes:
#   srun --jobid=<held> --overlap --nodes=2 --ntasks-per-node=1 bash scripts/run_cache_v22.sh
set -uo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export ATTN_BACKEND=sdpa TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

NODE=${NODE_OVERRIDE:-${SLURM_NODEID:-0}}
NNODES=${NNODES_OVERRIDE:-${SLURM_NNODES:-2}}
NSHARD=$((NNODES * 8))
MANI=/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_all.jsonl
OUT=/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1
CK_SRC=/fsx/sfr/weikaih/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000
CK_LOCAL=/dev/shm/v22ckpt
LOGD=runs/cache_logs/v22cache; mkdir -p "$LOGD"
echo "[v22cache] node=$NODE/$NNODES nshard=$NSHARD out=$(basename $OUT)"

# stage ckpt to node-local RAM ONCE (8 shards cold-loading 5.5GB from Lustre = ~47min each;
# from /dev/shm = seconds). Exclude DeepSpeed optimizer / rng states (huge + unneeded).
echo "[v22cache] node=$NODE staging ckpt → $CK_LOCAL"
mkdir -p "$CK_LOCAL"
rsync -a --exclude 'rng_state_*.pth' --exclude 'global_step*' --exclude '*.optim*' \
  --exclude 'latest' "$CK_SRC/" "$CK_LOCAL/" 2>&1 | tail -1
echo "[v22cache] node=$NODE ckpt staged: $(du -sh $CK_LOCAL 2>/dev/null | awk '{print $1}')"

echo "===== Phase V (v2.2 encoder) node=$NODE ====="
for g in $(seq 0 7); do
  GS=$((NODE * 8 + g))
  CUDA_VISIBLE_DEVICES=$g python scripts/build_vlm_cache_v22.py \
    --manifests "$MANI" --out_root "$OUT" --vlm "$CK_LOCAL" --shard "$GS" --num_shards "$NSHARD" \
    > "$LOGD/v_$GS.log" 2>&1 &
done
wait
echo "[v22cache] node=$NODE Phase V done"

echo "===== Phase D (DINOv3) node=$NODE ====="
for g in $(seq 0 7); do
  GS=$((NODE * 8 + g))
  CUDA_VISIBLE_DEVICES=$g python scripts/build_dino_cache.py \
    --manifest "$MANI" --out_root "$OUT" --crop_to_object 1 --max_views 1 \
    --shard "$GS" --num_shards "$NSHARD" > "$LOGD/d_$GS.log" 2>&1 &
done
wait
echo "[v22cache] node=$NODE Phase D done"

echo "===== Phase M (merge, CPU) node=$NODE ====="
for g in $(seq 0 7); do
  GS=$((NODE * 8 + g))
  python scripts/merge_vd_cache.py --root "$OUT" --manifest "$MANI" \
    --max_views 1 --shard "$GS" --num_shards "$NSHARD" > "$LOGD/m_$GS.log" 2>&1 &
done
wait
echo "[v22cache] node=$NODE ALL DONE"
