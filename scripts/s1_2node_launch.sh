#!/bin/bash
# Per-NODE launcher for 2-node S1 connector warmup. (1) build full MDS to local NVMe if
# absent, (2) /fsx file-barrier so both nodes start torchrun together (decouples build
# timing from rdzv), (3) multi-node torchrun S1. eff BS 256 = bs2 × ga8 × 16 GPU.
set -uo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export ATTN_BACKEND=flash_attn TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=x
export NCCL_TIMEOUT=3600
MDS=/opt/dlami/nvme/weikaih_mds
NID="${SLURM_NODEID:-0}"; NN="${SLURM_NNODES:-1}"
BAR=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/s1bar_${SLURM_JOB_ID}

echo "[node $NID/$NN @ $(hostname)] MDS check ..."
if [ ! -f "$MDS/index.json" ]; then
  echo "[node $NID] building full MDS (~70min) ..."
  python3 scripts/build_mds.py --out "$MDS" --procs 40 --size-limit 256mb 2>&1 | tail -3
fi
python3 -c "from streaming import StreamingDataset; print('[node $NID] MDS ready:', len(StreamingDataset(local='$MDS', shuffle=False, batch_size=1)))"

# file-barrier: every node signals ready, then waits for all
mkdir -p "$BAR"; touch "$BAR/ready_$NID"
echo "[node $NID] at barrier, waiting for $NN nodes ..."
for i in $(seq 1 360); do [ "$(ls "$BAR" | wc -l)" -ge "$NN" ] && break; sleep 10; done

echo "[node $NID] launching torchrun (master=$MASTER_ADDR:$MASTER_PORT) ..."
torchrun --nnodes="$NN" --node-rank="$NID" --nproc-per-node=8 \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  train_native.py --vlm_model Qwen/Qwen3.5-2B --mds_root "$MDS" \
  --build_vlm False --fuse_dino False \
  --build_slat True --ss_only False --freeze_vlm True --flow_tune none --distill_dino False \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --max_slat_tokens 8192 --elastic_slat True --elastic_target_ratio 0.75 \
  --output_dir runs/s1_2node --max_steps 3000 --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --warmup_steps 100 --weight_decay 0.01 \
  --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --ema_decay 0.9999 --optim adamw_torch_fused \
  --logging_steps 5 --save_steps 500 --save_total_limit 4 --report_to none \
  --deepspeed configs/deepspeed_zero1.json --ignore_data_skip True --dataloader_num_workers 0 2>&1 | grep -vE "load failed|resampling"
