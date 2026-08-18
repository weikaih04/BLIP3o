#!/bin/bash
# Per-node inner for the 2-node debug — called via `srun --nodes=2 --ntasks-per-node=1` from a HELD
# allocation (hold_2node.sbatch) OR directly. Needs MASTER_ADDR/MASTER_PORT + EFA env in the
# environment (the caller/hold job exports them; srun --overlap propagates). SLURM_NODEID → node-rank.
# Tests full-mode (--mds_root, standard global sharding, NO node-local patch) + EFA + compile off.
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
# self-contained EFA/NCCL env (so a bare `srun ... bash dbg_2node_inner.sh` works; only MASTER_ADDR/PORT needed from caller)
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_ASYNC_ERROR_HANDLING=1
export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn_3}" FUSED_MODULATE="${FUSED_MODULATE:-1}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=x
_IF="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$_IF}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$_IF}"
echo "[inner] node=$(hostname) NODE_RANK=${SLURM_NODEID:-0} MASTER=${MASTER_ADDR}:${MASTER_PORT} iface=${NCCL_SOCKET_IFNAME}"
torchrun --nnodes=2 --node-rank="${SLURM_NODEID:-0}" --nproc-per-node=8 \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  train_native.py --vlm_model Qwen/Qwen3.5-2B --mds_root /fsx/sfr/weikaih/3dgen/data/_mini_mds_v3 \
  --build_vlm False --fuse_dino False \
  --build_slat True --ss_only False --freeze_vlm True --flow_tune none --distill_dino False \
  --compile_ss_flow False \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --max_slat_tokens 8192 --elastic_slat True --elastic_target_ratio 0.75 \
  --output_dir runs/hold_dbg --max_steps 25 --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --warmup_steps 100 --weight_decay 0.01 \
  --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --ema_decay 0.9999 --optim adamw_torch_fused \
  --logging_steps 1 --save_steps 9999 --report_to none \
  --deepspeed configs/deepspeed_zero1.json --ignore_data_skip True --dataloader_num_workers 0 2>&1 | grep -vE "resampling|load failed"
