#!/bin/bash
# 2-node NCCL test WITH zhiyuan's EFA setup (the missing piece). mini-MDS full-mode (no staging).
# If this trains past step ~15 (where 16976 hung) → EFA fix works → real 2-node S1 is unblocked.
#SBATCH --job-name=dbg_2node
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --nodes=2
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=1:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/dbg_2node-%j.out
# MASTER as IPv4 (zhiyuan: hostname can socket-timeout on some nodes)
head="$(scontrol show hostnames "$SLURM_NODELIST" | head -1)"
export MASTER_ADDR="$(getent ahostsv4 "$head" 2>/dev/null | awk '{print $1; exit}')"
[ -z "$MASTER_ADDR" ] && export MASTER_ADDR="$(echo "$head" | sed -E 's/^ip-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+).*/\1.\2.\3.\4/')"
export MASTER_PORT="$(( 20000 + SLURM_JOB_ID % 20000 ))"
# ---- cross-node NCCL over EFA (the fix) ----
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
_IFACE="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
export NCCL_SOCKET_IFNAME="$_IFACE" GLOO_SOCKET_IFNAME="$_IFACE"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=x
export ATTN_BACKEND=flash_attn_3 FUSED_MODULATE=1
echo "[dbg] MASTER=$MASTER_ADDR:$MASTER_PORT iface=$_IFACE"
MDS=/fsx/sfr/weikaih/3dgen/data/_mini_mds_v3
srun --ntasks=2 --ntasks-per-node=1 bash -c '
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH"
torchrun --nnodes=2 --node-rank="$SLURM_NODEID" --nproc-per-node=8 \
  --master-addr="'"$MASTER_ADDR"'" --master-port="'"$MASTER_PORT"'" \
  train_native.py --vlm_model Qwen/Qwen3.5-2B --mds_root "'"$MDS"'" \
  --build_vlm False --fuse_dino False \
  --build_slat True --ss_only False --freeze_vlm True --flow_tune none --distill_dino False \
  --compile_ss_flow False \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --max_slat_tokens 4096 --elastic_slat True --elastic_target_ratio 0.75 \
  --output_dir runs/dbg_2node --max_steps 20 --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --warmup_steps 100 --weight_decay 0.01 \
  --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --ema_decay 0.9999 --optim adamw_torch_fused \
  --logging_steps 1 --save_steps 9999 --report_to none \
  --deepspeed configs/deepspeed_zero1.json --ignore_data_skip True --dataloader_num_workers 0' 2>&1 | grep -vE "resampling|load failed"
