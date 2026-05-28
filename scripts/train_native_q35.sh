#!/usr/bin/env bash
# Native-VLM continuous variant — Qwen3.5-2B backbone (no discrete).
# See QWEN35_VLM_DESIGN.md, OPTIMIZATIONS.md. ENV: activate `blip3o_trellis_qwen35`.
#
# REPO POLICY: NO OFFLOAD. We never use DeepSpeed CPU/NVMe offload to "make
# something fit". If a config doesn't fit, scale GPUs / cut batch / freeze more
# — don't slow training 3-4× to hide the real memory pressure. train_native.py
# hard-refuses any --deepspeed config that sets `offload_optimizer.device !=
# "none"` or any `offload_param.*`. The zero2_offload config file is kept on
# disk for reference but is unusable through our trainer.
# (Cf. OPTIMIZATIONS.md §"No-offload policy".)
#
# Usage: bash scripts/train_native_q35.sh [MODE=ss|512] [NPROC=?] [DATA_PATH=...]
#
# Terminology (matches user vocabulary, fixed 2026-05-28):
#   ss         → SS flow only                          (513 M trainable)
#   512        → SS + Shape SLAT 512 + Tex SLAT 512    (3.9 B trainable)
#                NOT real "cascade" — real cascade = 512 → 1024 ft progression,
#                which is not coded yet (would need a 1024-ft entry mode here).
#
# Resource floors (no-offload policy, see OPTIMIZATIONS.md):
#   ss   → any NPROC ≥ 1 OK
#   512  → NPROC ≥ 4 required (script hard-errors below otherwise)
set -euo pipefail
MODE="${1:-ss}"
NPROC="${2:-1}"
DATA_PATH="${3:-data/overfit/imgtext.jsonl}"
cd "$(dirname "$0")/.."

export PATH="$(dirname "$(command -v python)"):$PATH"

case "$MODE" in
  ss)
    BUILD_SLAT=False; SS_ONLY=True
    ;;
  512)
    if [ "${NPROC}" -lt 4 ]; then
      echo "[train_native_q35.sh] '512' stage (SS + SLAT 512) needs NPROC>=4 without offload (got ${NPROC})." >&2
      echo "  Use 'ss' mode on smaller setups, or run 512 on >=4 GPUs." >&2
      exit 1
    fi
    BUILD_SLAT=True; SS_ONLY=False
    ;;
  cascade)
    echo "[train_native_q35.sh] MODE='cascade' was renamed to '512' on 2026-05-28 to match" >&2
    echo "  upstream TRELLIS terminology (real cascade = 512 → 1024 ft progression, not coded yet)." >&2
    echo "  Use MODE=512 instead." >&2
    exit 1
    ;;
  *)
    echo "[train_native_q35.sh] unknown MODE='${MODE}'. Use 'ss' or '512'." >&2
    exit 1
    ;;
esac
DS=configs/deepspeed_zero2.json   # always no-offload; see policy above

torchrun --nproc_per_node="${NPROC}" train_native.py \
  --vlm_model "Qwen/Qwen3.5-2B" \
  --data_path "${DATA_PATH}" \
  --freeze_vlm True \
  --build_slat "${BUILD_SLAT}" \
  --ss_only "${SS_ONLY}" \
  --num_cond_views 1 \
  --output_dir "runs/native_q35_mode${MODE}" \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_steps 300 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --report_to none \
  --optim adamw_torch_fused \
  --compile_ss_flow True \
  --compile_mode default \
  --deepspeed "${DS}"
