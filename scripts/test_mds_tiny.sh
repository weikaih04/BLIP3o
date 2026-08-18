#!/bin/bash
set -uo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export ATTN_BACKEND=flash_attn TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=x CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 train_native.py \
  --vlm_model Qwen/Qwen3.5-2B --mds_root /opt/dlami/nvme/weikaih_mds_e2e \
  --build_vlm False --fuse_dino True --dino_drop_prob 0.3 \
  --train_stages shape --build_slat True --ss_only False --compile_ss_flow False \
  --elastic_slat True --elastic_target_ratio 0.1 --max_slat_tokens 8192 \
  --freeze_vlm True --flow_tune last20 \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --output_dir runs/mds_tiny --max_steps 4 --bf16 True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --warmup_steps 100 --max_grad_norm 1.0 --ema_decay 0.998 \
  --logging_steps 1 --save_steps 100000 --report_to none \
  --ddp_find_unused_parameters False --dataloader_num_workers 0
