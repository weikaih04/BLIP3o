#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ① S1 — connector warm-up (the FIRST link of the curriculum).
# Ground truth: experiment/qwen35_trellis2_training/train_v3_stage1_nokd_8g.yaml
# (commit 96a0b26). ONE job: connector-only (all 3 flows FROZEN, Qwen frozen),
# LIVE I1 VLM (no cache, no fusion), pure flow-matching loss (NO KD).
# Produces checkpoint-3000 → fed as --init_from_checkpoint to the fusion split.
#
# H200 override vs ground truth: elastic_target_ratio 0.1 → 0.75 (141G headroom).
# EMA 0.9999 here (NOT 0.998 — that's the split stages).
#
# Usage: bash scripts/train_connector_warmup.sh [NPROC=8]
#   env: PER_GPU_BS(2) ELASTIC_RATIO(0.75) MAX_STEPS(3000) LR(1e-4) RUN_TAG
#   Holds eff BS 256: ga = 256 / (PER_GPU_BS × NPROC)  → bs2×nproc8 = ga16.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
NPROC="${1:-8}"
PER_GPU_BS="${PER_GPU_BS:-2}"
ELASTIC_RATIO="${ELASTIC_RATIO:-0.75}"
MAX_STEPS="${MAX_STEPS:-3000}"
LR="${LR:-1e-4}"
MIX="${MIX:-configs/mix_i1_only_local.yaml}"   # local ready_v2 manifest (ground truth = mix_i1_only.yaml → weka)
REPORT_TO="${REPORT_TO:-wandb}"                  # set REPORT_TO=none for a key-less first run
# BUILD_VLM=True  → live VLM (MIX=mix_i1_only_local.yaml, the GT Stage-1 path)
# BUILD_VLM=False → CACHED cond, skip loading the 2B VLM (use MIX=mix_i1_cached_local.yaml)
BUILD_VLM="${BUILD_VLM:-True}"
command -v torchrun >/dev/null 2>&1 || source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd "$(dirname "$0")/.."
export PATH="$(dirname "$(command -v python)"):$PATH"

EFF_BS=256
GA=$(( EFF_BS / (PER_GPU_BS * NPROC) ))
[ $(( PER_GPU_BS * NPROC * GA )) -eq $EFF_BS ] || {
  echo "[connector-warmup] PER_GPU_BS($PER_GPU_BS) × NPROC($NPROC) must divide 256" >&2; exit 1; }
OUT="runs/p1_connector_warmup${RUN_TAG:+_$RUN_TAG}"
# DEEPSPEED=none → plain DDP (no ZeRO-2). For connector-only (3.1M trainable) ZeRO shards
# nothing → pure overhead; profiling showed ~21s/step of it. Default keeps zero2.
DEEPSPEED="${DEEPSPEED:-configs/deepspeed_zero2.json}"
if [ "$DEEPSPEED" = "none" ]; then DS_FLAG=""; else DS_FLAG="--deepspeed $DEEPSPEED"; fi
# NOTE (2026-06-27): tried dataloader workers + pin + persistent + prefetch to overlap data —
# it made it SLOWER (workers steal CPU from the dispatch-bound main thread). PROFILE_STEP showed
# prep(sparse H2D)=2ms, data=tiny; the cost is the CPU-dispatch-bound fwd+bwd. Keep workers=0.
DL_OPT=""
# MDS_ROOT (MosaicML Streaming) overrides the manifest mixture + forces cached cond (build_vlm
# False). S1 warmup is connector-only on the Qwen cond; FUSE_DINO defaults False (GT no-fusion).
if [ -n "${MDS_ROOT:-}" ]; then
  DATA_FLAG="--mds_root $MDS_ROOT --fuse_dino ${FUSE_DINO:-False}"; BUILD_VLM=False; WORKERS=0
else
  DATA_FLAG="--mixture_config $MIX"
fi
echo "[connector-warmup] nproc=$NPROC bs=$PER_GPU_BS ga=$GA (eff $EFF_BS) ds=$DEEPSPEED out=$OUT ${MDS_ROOT:+mds=$MDS_ROOT}"

torchrun --nproc_per_node="$NPROC" train_native.py \
  --vlm_model Qwen/Qwen3.5-2B \
  ${DATA_FLAG} \
  --build_vlm "$BUILD_VLM" \
  --build_slat True --ss_only False \
  --freeze_vlm True --flow_tune none --distill_dino False \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --max_slat_tokens 4096 --elastic_slat "${ELASTIC_SLAT:-True}" --elastic_target_ratio "$ELASTIC_RATIO" \
  --compile_ss_flow "${COMPILE_SS:-False}" --compile_mode default \
  --output_dir "$OUT" \
  --max_steps "$MAX_STEPS" --bf16 True \
  --per_device_train_batch_size "$PER_GPU_BS" --gradient_accumulation_steps "$GA" \
  --learning_rate "$LR" --warmup_steps 100 --weight_decay 0.01 \
  --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 \
  --ema_decay 0.9999 \
  --optim adamw_torch_fused \
  --logging_steps 5 --save_steps 500 --save_total_limit 4 \
  --report_to "$REPORT_TO" ${DS_FLAG} ${DDP_FLAGS:-} ${DL_OPT} \
  --ignore_data_skip True --dataloader_num_workers "${WORKERS:-0}"
