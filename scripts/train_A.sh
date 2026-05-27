#!/usr/bin/env bash
# Setup A — no codebook, full hidden, text-only CE + flow MSE
# MolmoAct2 / BLIP3-o original style.
#
# Usage: bash scripts/train_A.sh [NPROC_PER_NODE=8] [DATA_PATH=data/index_setup_A.jsonl]
set -euo pipefail
NPROC="${1:-8}"
DATA_PATH="${2:-data/index_setup_A.jsonl}"

cd "$(dirname "$0")/.."

torchrun --nproc_per_node="${NPROC}" train.py \
  --model_name_or_path "BLIP3o/BLIP3o-NEXT-Pretrain-3B" \
  --diffusion_name_or_path "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers" \
  --mm_tunable_parts "mm_language_model,mm_diffusion,mm_embedding" \
  --num_image_tokens 0 \
  --num_scale_tokens 0 \
  --mm_use_im_start_end false \
  --mm_vision_select_layer -1 \
  --mm_patch_merge_type "flat" \
  --data_path "${DATA_PATH}" \
  --dataset_cls "tr2_3d" \
  --use_codebook false \
  --num_views 4 \
  --is_multimodal true \
  --image_aspect_ratio square \
  --setup_name A \
  --cond_slice full \
  --detach_cond true \
  --cond_max_length 8192 \
  --flow_weight 1.0 \
  --flow_stage_weights "ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0" \
  --logitnorm_mean 1.0 \
  --logitnorm_std 1.0 \
  --output_dir runs/setup_A \
  --deepspeed configs/deepspeed_zero2.json \
  --bf16 true \
  --gradient_checkpointing true \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_steps 50000 \
  --save_steps 2000 \
  --logging_steps 50 \
  --save_total_limit 3 \
  --remove_unused_columns false \
  --attn_implementation flash_attention_2 \
  --model_max_length 8192
