#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DEPRECATED (2026-05-28) — Qwen3-VL-2B backbone is no longer supported.
# Use scripts/train_native_q35.sh (Qwen3.5-2B) instead.
# Body kept commented for reference; see Memory: project_qwen35_vlm_variant.md
# (Option α validated → Qwen3.5-2B is the chosen backbone).
# ─────────────────────────────────────────────────────────────────────────────
# Native-VLM continuous variant — Qwen3-VL-2B backbone (no discrete).
# See QWEN35_VLM_DESIGN.md. ENV: activate `blip3o_trellis` (transformers 4.57.6 has qwen3_vl).
#
# Usage: bash scripts/train_native_q3vl.sh [MODE=ss|cascade] [NPROC=1] [DATA_PATH=data/overfit/imgtext.jsonl]
#   ss      → SS-flow only (light; fits 1 GPU no-offload)
#   cascade → SS+Shape+Tex (3.9B trainable; needs ZeRO-2 CPU-offload)
echo "[DEPRECATED] train_native_q3vl.sh — Qwen3-VL backbone is no longer supported."
echo "             Use scripts/train_native_q35.sh (Qwen3.5-2B) instead."
exit 1

# set -euo pipefail
# MODE="${1:-ss}"
# NPROC="${2:-1}"
# DATA_PATH="${3:-data/overfit/imgtext.jsonl}"
# cd "$(dirname "$0")/.."
#
# # Ensure cpu_adam's JIT compiler (ninja) is on PATH for the offload optimizer.
# export PATH="$(dirname "$(command -v python)"):$PATH"
#
# if [ "$MODE" = "cascade" ]; then
#   BUILD_SLAT=True; SS_ONLY=False; DS=configs/deepspeed_zero2_offload.json
# else
#   BUILD_SLAT=False; SS_ONLY=True;  DS=configs/deepspeed_zero2.json
# fi
#
# torchrun --nproc_per_node="${NPROC}" train_native.py \
#   --vlm_model "Qwen/Qwen3-VL-2B-Instruct" \
#   --data_path "${DATA_PATH}" \
#   --freeze_vlm True \
#   --build_slat "${BUILD_SLAT}" \
#   --ss_only "${SS_ONLY}" \
#   --num_cond_views 1 \
#   --output_dir "runs/native_q3vl_${MODE}" \
#   --bf16 True \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 1e-4 \
#   --max_steps 300 \
#   --logging_steps 10 \
#   --save_steps 100 \
#   --save_total_limit 2 \
#   --report_to none \
#   --deepspeed "${DS}"
# # Unfreeze VLM (D2): add --freeze_vlm False + use the offload config + ≥4 GPUs (or VLM-LoRA).
