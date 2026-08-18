#!/bin/bash
# Single-node S1 step-time probe — sidesteps the 2-node cross-node NCCL hang to (a) validate
# training is sound with all fixes (compile off, EOFError-skip) and (b) measure the real S1
# opt-step time. Uses shards 0+2 already on ip-10-0-193-194's NVMe (no staging). bs2 ga8 = the
# same per-GPU work as the 2-node config, so opt-step ≈ the clean 2-node step (minus cross-node AR).
#SBATCH --job-name=s1_1node
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --exclusive
#SBATCH --nodelist=ip-10-0-193-194
#SBATCH --time=1:30:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/s1_1node-%j.out
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=x
export ATTN_BACKEND=flash_attn_3 FUSED_MODULATE=1
D=/opt/dlami/nvme/weikaih_mds_v3_shards
torchrun --nproc_per_node=8 train_native.py --vlm_model Qwen/Qwen3.5-2B \
  --mds_shards "$D/shard0,$D/shard2" \
  --build_vlm False --fuse_dino False \
  --build_slat True --ss_only False --freeze_vlm True --flow_tune none --distill_dino False \
  --compile_ss_flow False \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --max_slat_tokens 8192 --elastic_slat True --elastic_target_ratio 0.75 \
  --output_dir runs/s1_1node_probe --max_steps 60 --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --warmup_steps 100 --weight_decay 0.01 \
  --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --ema_decay 0.9999 --optim adamw_torch_fused \
  --logging_steps 1 --save_steps 9999 --save_total_limit 1 --report_to none \
  --deepspeed configs/deepspeed_zero1.json --ignore_data_skip True --dataloader_num_workers 0 2>&1 | grep -vE "load failed|resampling"
