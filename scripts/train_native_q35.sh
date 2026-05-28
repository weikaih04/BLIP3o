#!/usr/bin/env bash
# Native-VLM continuous variant — Qwen3.5-2B backbone (no discrete).
# See QWEN35_VLM_DESIGN.md. ENV: activate `blip3o_trellis_qwen35`
# (cp -a clone of blip3o_trellis + transformers==5.2.0, which registers qwen3_5).
# Byte-identical to train_native_q3vl.sh except --vlm_model (backbone A/B).
#
# Usage: bash scripts/train_native_q35.sh [MODE=ss|cascade] [NPROC=1] [DATA_PATH=data/overfit/imgtext.jsonl]
set -euo pipefail
MODE="${1:-ss}"
NPROC="${2:-1}"
DATA_PATH="${3:-data/overfit/imgtext.jsonl}"
cd "$(dirname "$0")/.."

export PATH="$(dirname "$(command -v python)"):$PATH"   # ninja for cpu_adam

if [ "$MODE" = "cascade" ]; then
  BUILD_SLAT=True; SS_ONLY=False; DS=configs/deepspeed_zero2_offload.json
else
  BUILD_SLAT=False; SS_ONLY=True;  DS=configs/deepspeed_zero2.json
fi

torchrun --nproc_per_node="${NPROC}" train_native.py \
  --vlm_model "Qwen/Qwen3.5-2B" \
  --data_path "${DATA_PATH}" \
  --freeze_vlm True \
  --build_slat "${BUILD_SLAT}" \
  --ss_only "${SS_ONLY}" \
  --num_cond_views 1 \
  --output_dir "runs/native_q35_${MODE}" \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_steps 300 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --report_to none \
  --deepspeed "${DS}"
