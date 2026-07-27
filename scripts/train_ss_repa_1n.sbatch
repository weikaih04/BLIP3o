#!/bin/bash
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --time=1-00:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/ss_repa_%x_%j.out
# SS-REPA pilot arm — SINGLE node (so both arms fit the 2-node low-pri user cap in parallel).
# Continue-train SS from s3_ss_im_mds/ckpt-10000 on the 16k REPA-subset MDS (aux coverage 100%),
# IM_VIEW_SUBSET=1 both arms. REPA arm: REPA_ROOT set → --repa_root/--repa_coeff via the same
# env pattern as train_im_mds_2n.sbatch. Control arm: REPA_ROOT unset — ONLY difference.
#   sbatch --job-name=repa    --export=ALL,RUN_TAG=ss_repa16k,REPA_ROOT=/fsx/home/weikai.huang/3dgen/im_probe/vggt16_targets  scripts/train_ss_repa_1n.sbatch
#   sbatch --job-name=repactl --export=ALL,RUN_TAG=ss_repactl16k  scripts/train_ss_repa_1n.sbatch
set -u
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
INIT="${INIT:-runs/s3_ss_im_mds/checkpoint-10000}"
MANI="${MANI:-/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_repa16k.jsonl}"
NV="${NV:-/opt/dlami/nvme/weikaih_im_mds_repa16k}"
STEPS="${STEPS:-8000}"
RUN_TAG="${RUN_TAG:?set RUN_TAG (ss_repa16k | ss_repactl16k)}"
REPA_FLAGS=""
if [ -n "${REPA_ROOT:-}" ]; then
  # v2 grid knobs: REPA_ZDIM (2049 for raw-2048 targets), REPA_DEPTH (5 = REPA-early tap),
  # REPA_COEFF (λ). All default via ${VAR:-…} so v1 (32-d) still works if unset.
  REPA_FLAGS=" --repa_root $REPA_ROOT --repa_coeff ${REPA_COEFF:-0.5}"
  REPA_FLAGS="$REPA_FLAGS --repa_zdim ${REPA_ZDIM:-33} --repa_depth ${REPA_DEPTH:-0}"
fi
echo "[ss_repa_1n] tag=$RUN_TAG init=$INIT steps=$STEPS repa=${REPA_ROOT:-off}"

source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
D=$NV/shard0
# node-local 16k-subset IM MDS (1 shard = whole subset; ~2 min build)
if [ ! -f "$D/index.json" ]; then
  rm -rf "$D"
  COND_MODE=im python scripts/build_mds.py --out "$D" \
    --manifest "$MANI" --shard 0 --num-shards 1 --procs 24 \
    > "runs/cache_logs/repa16k_mds_${RUN_TAG}.log" 2>&1
fi
echo "[ss_repa_1n] shard ready"
# /dev/shm is not writable on some xgen-mm nodes (Permission denied) — put triton/inductor/tmp
# caches on node-local NVMe instead (COMPILE_SS=0 so these barely used, but TMPDIR must be writable).
CACHE_BASE="/opt/dlami/nvme/repacache_${SLURM_JOB_ID:-0}"
mkdir -p "$CACHE_BASE/ind" "$CACHE_BASE/tri" "$CACHE_BASE/tmp" 2>/dev/null \
  || { CACHE_BASE="$HOME/.repacache_${SLURM_JOB_ID:-0}"; mkdir -p "$CACHE_BASE/ind" "$CACHE_BASE/tri" "$CACHE_BASE/tmp"; }
env DEEPSPEED=none DINO_DROP=0.1 MAX_STEPS="$STEPS" RUN_TAG="$RUN_TAG" REPORT_TO=none \
    IM_VIEW_SUBSET=1 REPA_ROOT="${REPA_ROOT:-}" \
    TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/ind" TRITON_CACHE_DIR="$CACHE_BASE/tri" TMPDIR="$CACHE_BASE/tmp" \
    VIEW_EMBED_SCALE="${VIEW_EMBED_SCALE:-1.0}" REPA_COEFF_HI="${REPA_COEFF_HI:-}" REPA_WARMUP_STEPS="${REPA_WARMUP_STEPS:-2000}" \
    PER_GPU_BS="${PER_GPU_BS:-8}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    COMPILE_SS=0 MDS_WORKERS=8 NNODES=1 MDS_SHARDS=$D \
    EXTRA_ARGS="--save_steps 2000 --save_total_limit 8 --cond_adapter mlp$REPA_FLAGS${VIEW_EMBED_MODE:+ --view_embed_mode $VIEW_EMBED_MODE}" \
    bash scripts/train_native_split.sh ss s3 "$INIT" 8 \
    > "runs/cache_logs/${RUN_TAG}_train.log" 2>&1
echo "=== ss_repa_1n $RUN_TAG DONE ==="
