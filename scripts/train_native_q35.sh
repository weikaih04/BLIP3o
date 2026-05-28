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
# Usage: bash scripts/train_native_q35.sh [MODE=ss|cascade] [NPROC=?] [DATA_PATH=...]
#   SS-only   (513 M trainable):  any NPROC ≥ 1 OK.
#   cascade   (3.9 B trainable):  requires NPROC ≥ 4 without offload — the script
#                                  hard-errors below if you ask for fewer.
set -euo pipefail
MODE="${1:-ss}"
NPROC="${2:-1}"
DATA_PATH="${3:-data/overfit/imgtext.jsonl}"
cd "$(dirname "$0")/.."

export PATH="$(dirname "$(command -v python)"):$PATH"

if [ "$MODE" = "cascade" ]; then
  if [ "${NPROC}" -lt 4 ]; then
    echo "[train_native_q35.sh] cascade needs NPROC>=4 without offload (got ${NPROC})." >&2
    echo "  Use 'ss' mode on smaller setups, or run cascade on >=4 GPUs." >&2
    exit 1
  fi
  BUILD_SLAT=True; SS_ONLY=False
else
  BUILD_SLAT=False; SS_ONLY=True
fi
DS=configs/deepspeed_zero2.json   # always no-offload; see policy above

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
  --optim adamw_torch_fused \
  --compile_ss_flow True \
  --compile_mode default \
  --deepspeed "${DS}"
