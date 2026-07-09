#!/bin/bash
# Per-NODE launcher for ready_v3 multi-node S1/S2 (node-SHARDED MDS, 1/2/4 nodes).
# (1) stage: build ONLY this node's owned shards (k%NNODES==NODEID) to local NVMe from /fsx
#     source — /fsx (92% full) can't persist a 6TB MDS, and replicating the full set to every
#     node wastes ~1.9h+6TB each. 4 nodes→1 shard/node(~30min); 2 nodes→2/node(~1h).
# (2) /fsx file-barrier so all nodes start torchrun together (decouples build time from rdzv).
# (3) torchrun with --mds_shards = this node's shard dirs → StreamingImageTo3D shards node-locally.
set -uo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn_3}" FUSED_MODULATE="${FUSED_MODULATE:-1}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="${WANDB_API_KEY:-f773908953fc7bea7008ae1cf3701284de1a0682}"
export NCCL_TIMEOUT=3600
# ---- cross-node NCCL/EFA env, lifted from zhiyuan's working multi-node setup (43_train_inner.sh).
# NOTE: EFA actually auto-loads via ldconfig (verified — NCCL picks NET/OFI+efa+RDMA on its own),
# so this is mostly explicit hygiene, NOT the proven hang fix. The parts that could still matter:
# NCCL_SOCKET_IFNAME pins the bootstrap NIC (avoids NCCL auto-picking a docker/bridge iface among
# the ~15 present), and NCCL_NET_GDR_LEVEL/ASYNC_ERROR_HANDLING. Runs per-node via srun. Harmless 1-node.
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
export FI_PROVIDER="${FI_PROVIDER:-efa}"
export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_ASYNC_ERROR_HANDLING=1
_IFACE="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
[ -z "$_IFACE" ] && for n in /sys/class/net/*; do [ -e "$n/device" ] && { _IFACE="$(basename "$n")"; break; }; done
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$_IFACE}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$_IFACE}"

DEST="${MDS_V3_ROOT:-/opt/dlami/nvme/weikaih_mds_v3_shards}"
NUM_SHARDS="${MDS_NUM_SHARDS:-4}"
NID=$(( ${NID_BASE:-0} + ${SLURM_NODEID:-0} )); NN="${NN_TOTAL:-${SLURM_NNODES:-1}}"
# barrier dir must be unique PER srun STEP (held-node pattern reuses one SLURM_JOB_ID across many
# sruns → stale ready_* files would let a later run sail through the barrier unsynchronized)
BAR="${BAR_DIR:-/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/s1v3bar_${SLURM_JOB_ID:-local}_${SLURM_STEP_ID:-0}}"
OUTDIR="${OUTDIR:-runs/s1_v3}"; MAX_STEPS="${MAX_STEPS:-3000}"

# (1) stage data. MDS_MODE=full → FOLLOW ZHIYUAN: every node builds the FULL MDS, StreamingDataset
# does STANDARD global node/rank sharding (no node-local patch — removes the multi-node hang suspect).
# MDS_MODE=shard (default) → node-local patch (each node only its 1/N).
MDS_MODE="${MDS_MODE:-shard}"
DS_FLAG=$([ "${DEEPSPEED:-zero1}" = "none" ] && echo "" || echo "--deepspeed configs/deepspeed_${DEEPSPEED:-zero1}.json")
MANI="${MANI:-/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v3/ready_v3_clean.jsonl}"
PY=/fsx/sfr/weikaih/miniconda3/envs/blip3o_trellis/bin/python
if [ "$MDS_MODE" = "full" ]; then
  FULL="$DEST/full"
  if [ ! -f "$FULL/index.json" ]; then
    echo "[node $NID] building FULL MDS (~1.9h, ~6TB) → $FULL"
    "$PY" scripts/build_mds.py --out "$FULL" --manifest "$MANI" --procs 32
  else echo "[node $NID] FULL MDS present → skip build"; fi
  DATA_FLAG="--mds_root $FULL"
  "$PY" -c "from streaming import StreamingDataset; print('[node $NID] full MDS samples:', len(StreamingDataset(local='$FULL', shuffle=False, batch_size=1)))"
else
  MDS_NUM_SHARDS="$NUM_SHARDS" MDS_V3_ROOT="$DEST" bash scripts/stage_mds_v3.sh
  OWNED=""
  for K in $(seq 0 $((NUM_SHARDS-1))); do
    [ $((K % NN)) -eq "$NID" ] || continue
    OWNED="${OWNED:+$OWNED,}$DEST/shard$K"
  done
  echo "[node $NID/$NN @ $(hostname)] owned shards: $OWNED"
  DATA_FLAG="--mds_shards $OWNED"
fi

# (2) file-barrier: every node signals ready, then waits for all
mkdir -p "$BAR"; touch "$BAR/ready_$NID"
echo "[node $NID] at barrier, waiting for $NN nodes ..."
for i in $(seq 1 720); do [ "$(ls "$BAR" | wc -l)" -ge "$NN" ] && break; sleep 10; done

# (3) multi-node torchrun (node-local data sharding via --mds_shards)
echo "[node $NID] launching torchrun (master=${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500}) ..."
torchrun --nnodes="$NN" --node-rank="$NID" --nproc-per-node=8 \
  --master-addr="${MASTER_ADDR:-localhost}" --master-port="${MASTER_PORT:-29500}" \
  train_native.py --vlm_model Qwen/Qwen3.5-2B $DATA_FLAG \
  --build_vlm False --fuse_dino False \
  --build_slat True --ss_only False --freeze_vlm True --flow_tune none --distill_dino False \
  --compile_ss_flow False \
  `# S1 FREEZES all flows (flow_tune none) → compiling the frozen ss_flow makes Dynamo recompile`\
  `# every step on a 'requires_grad mismatch' guard (cache thrash); multi-node that desyncs ranks`\
  `# → NCCL allreduce timeout → SIGABRT (job 16976 died at step 15 this way). Frozen flow gains`\
  `# nothing from compile, so disable it here. (compile stays ON for the trainable split ss stage.)`\
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --max_slat_tokens "${MAX_SLAT:-8192}" --elastic_slat True --elastic_target_ratio "${ELASTIC_RATIO:-0.75}" \
  --output_dir "$OUTDIR" --max_steps "$MAX_STEPS" --bf16 True \
  --per_device_train_batch_size "${PER_GPU_BS:-2}" --gradient_accumulation_steps "${GA:-8}" \
  --learning_rate 1e-4 --warmup_steps 100 --weight_decay 0.01 \
  --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --ema_decay 0.9999 --optim adamw_torch_fused \
  --logging_steps 5 --save_steps 500 --save_total_limit 4 --report_to none \
  ${DS_FLAG} --ignore_data_skip True --dataloader_num_workers 0 2>&1 | grep -vE "load failed|resampling"
