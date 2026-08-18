#!/bin/bash
# Step-time PROBE for all stages on a 2000-sample mini-MDS (no 1h staging). INIT=none
# (init weights don't affect step time). Each stage ~20 steps → read steady-state s/it.
# 8-GPU eff-BS 256; for 16-GPU halve the ga → ~half the opt-step (+ small allreduce).
#SBATCH --job-name=stage_probe
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=1:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/stage_probe-%j.out
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=x
export ATTN_BACKEND=flash_attn_3 FUSED_MODULATE=1
MDS=/fsx/sfr/weikaih/3dgen/data/_mini_mds_v3

echo "########## PROBE S1 (connector warmup · 3 flows together · bs2 ga16 eff256) ##########"
torchrun --nproc_per_node=8 train_native.py --vlm_model Qwen/Qwen3.5-2B --mds_root "$MDS" \
  --build_vlm False --fuse_dino False \
  --build_slat True --ss_only False --freeze_vlm True --flow_tune none --distill_dino False \
  --compile_ss_flow False \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --max_slat_tokens 8192 --elastic_slat True --elastic_target_ratio 0.75 \
  --output_dir runs/probe_s1 --max_steps 20 --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --warmup_steps 100 --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.95 \
  --max_grad_norm 1.0 --ema_decay 0.9999 --optim adamw_torch_fused \
  --logging_steps 1 --save_steps 9999 --save_total_limit 1 --report_to none \
  --deepspeed configs/deepspeed_zero1.json --ignore_data_skip True --dataloader_num_workers 0 2>&1 | grep -vE "load failed|resampling" || echo "[probe] S1 FAILED"

# S2 fusion: ss / shape / tex as the real split jobs (bs8 ga4 eff256, flow_tune last20, fuse_dino)
export REPORT_TO=none MAX_STEPS=20 MDS_ROOT="$MDS"
for STAGE in ss shape tex; do
  echo "########## PROBE S2-$STAGE (fusion · 1 flow · bs8 ga4 eff256 · flow_tune last20) ##########"
  bash scripts/train_native_split.sh "$STAGE" fusion none 8 2>&1 | grep -vE "load failed|resampling" || echo "[probe] S2-$STAGE FAILED"
done
echo "########## PROBE ALL DONE ##########"
