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
# Usage: bash scripts/train_native_q35.sh [MODE=ss|512] [NPROC=?] [DATA=...]
#   DATA defaults to configs/mix_3d_only.yaml (real multi-task 3D mixture over
#   ready_v1 80,076 assets). Pass a .yaml → mixture mode; pass a .jsonl →
#   legacy single-manifest (e.g. data/overfit/imgtext.jsonl for overfit smoke).
#
# Terminology (matches user vocabulary, fixed 2026-05-28):
#   ss         → SS flow only                          (513 M trainable)
#   512        → SS + Shape SLAT 512 + Tex SLAT 512    (3.9 B trainable)
#                NOT real "cascade" — real cascade = 512 → 1024 ft progression,
#                which is not coded yet (would need a 1024-ft entry mode here).
#
# Resource floors (no-offload policy, see OPTIMIZATIONS.md):
#   ss   → any NPROC ≥ 1 OK, per-gpu BS=2
#   512  → any NPROC ≥ 1 OK, per-gpu BS=2 (full cascade fits on 1×80G; the old
#          ">=4 GPU" floor is obsolete as of 2026-05-30)
set -euo pipefail
MODE="${1:-ss}"
NPROC="${2:-1}"
# 4th arg = data source. Default = the real multi-task 3D mixture over ready_v1
# (80,076 512-trainable assets; 3 tasks text/image/multi_image_to_3d).
#   - a path ending in .yaml  → --mixture_config (multi-task)
#   - any other path          → --data_path (legacy single-manifest, e.g. overfit)
DATA="${3:-configs/mix_3d_only.yaml}"
cd "$(dirname "$0")/.."

export PATH="$(dirname "$(command -v python)"):$PATH"

case "$MODE" in
  ss)
    # SS-only = 513M trainable. per-gpu BS=2 fits easily on 80G. Default 2.
    BUILD_SLAT=False; SS_ONLY=True; PER_GPU_BS="${PER_GPU_BS:-2}"
    ;;
  512)
    # Full 512 cascade (SS + Shape SLAT + Tex SLAT, 3.9B trainable) FITS ON 1 GPU
    # at per-gpu BS=2, no offload (verified 2026-05-30 — the earlier ">=4 GPU"
    # requirement is OBSOLETE; compile + fused AdamW + partial-FT cut the memory).
    # Override with PER_GPU_BS env if you hit OOM on a smaller card.
    BUILD_SLAT=True; SS_ONLY=False; PER_GPU_BS="${PER_GPU_BS:-2}"
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

# Route the 4th arg: .yaml → mixture (multi-task), else → single manifest.
case "$DATA" in
  *.yaml|*.yml) DATA_FLAG="--mixture_config ${DATA}" ;;
  *)            DATA_FLAG="--data_path ${DATA}" ;;
esac

torchrun --nproc_per_node="${NPROC}" train_native.py \
  --vlm_model "Qwen/Qwen3.5-2B" \
  ${DATA_FLAG} \
  --freeze_vlm True \
  --build_slat "${BUILD_SLAT}" \
  --ss_only "${SS_ONLY}" \
  --num_cond_views 1 \
  --output_dir "runs/native_q35_mode${MODE}" \
  --bf16 True \
  --per_device_train_batch_size "${PER_GPU_BS}" \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --adam_beta2 0.95 \
  --max_steps 300 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --report_to none \
  --optim adamw_torch_fused \
  --compile_ss_flow True \
  --compile_mode default \
  --deepspeed "${DS}"
