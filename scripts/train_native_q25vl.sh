#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DEPRECATED (2026-05-28) — Qwen2.5-VL-3B backbone is no longer supported.
# Use scripts/train_native_q35.sh (Qwen3.5-2B) instead.
# Body kept commented for reference.
# ─────────────────────────────────────────────────────────────────────────────
# Native-VLM continuous variant — Qwen2.5-VL-3B backbone (matches the encoder the
# latest OPEN Qwen-Image-Edit-2511 uses). No discrete tokens. See QWEN35_VLM_DESIGN.md.
# ENV: activate `blip3o_trellis` — transformers 4.57.6 ALREADY has qwen2_5_vl (no clone,
# same env as Qwen3-VL). Backbone-agnostic model: only --vlm_model differs from q3vl/q35.
#
# Usage: bash scripts/train_native_q25vl.sh [MODE=ss|cascade] [NPROC=1] [DATA_PATH] [FLOW_TUNE]
echo "[DEPRECATED] train_native_q25vl.sh — Qwen2.5-VL backbone is no longer supported."
echo "             Use scripts/train_native_q35.sh (Qwen3.5-2B) instead."
exit 1

# set -euo pipefail
# MODE="${1:-ss}"
# NPROC="${2:-1}"
# DATA_PATH="${3:-data/overfit/imgtext.jsonl}"
# FLOW_TUNE="${4:-last40}"          # last40 = no-offload-friendly partial-FT (mirrors blip3o ladder)
# cd "$(dirname "$0")/.."
#
# export PATH="$(dirname "$(command -v python)"):$PATH"   # ninja for cpu_adam (cascade)
#
# if [ "$MODE" = "cascade" ]; then
#   BUILD_SLAT=True; SS_ONLY=False; DS=configs/deepspeed_zero2_offload.json
# else
#   BUILD_SLAT=False; SS_ONLY=True;  DS=configs/deepspeed_zero2.json
# fi
#
# torchrun --nproc_per_node="${NPROC}" train_native.py \
#   --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \
#   --data_path "${DATA_PATH}" \
#   --freeze_vlm True \
#   --flow_tune "${FLOW_TUNE}" \
#   --build_slat "${BUILD_SLAT}" \
#   --ss_only "${SS_ONLY}" \
#   --num_cond_views 1 \
#   --output_dir "runs/native_q25vl_${MODE}" \
#   --bf16 True \
#   --per_device_train_batch_size 1 \
#   --learning_rate 1e-4 \
#   --max_steps 300 \
#   --logging_steps 10 \
#   --save_steps 100 \
#   --save_total_limit 2 \
#   --report_to none \
#   --deepspeed "${DS}"
