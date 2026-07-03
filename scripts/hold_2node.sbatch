#!/bin/bash
# 2-node HOLD + interactive debug allocation. When it lands it (1) auto-runs the full-mode+EFA
# 2-node debug once (so we get the fix answer even if nobody's watching), then (2) HOLDS the
# allocation (sleep) so you can srun/ssh in and iterate WITHOUT re-queuing each attempt.
# ⚠️ Holding idle GPUs violates the cluster ≥50%-util rule → run your debug promptly or it may be
#    flagged/killed. --time caps it at 6h so it can't hoard forever.
#SBATCH --job-name=distributed_training
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=2
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=6:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/hold_2n-%j.out
head="$(scontrol show hostnames "$SLURM_NODELIST" | head -1)"
export MASTER_ADDR="$(getent ahostsv4 "$head" 2>/dev/null | awk '{print $1; exit}')"
[ -z "$MASTER_ADDR" ] && export MASTER_ADDR="$(echo "$head" | sed -E 's/^ip-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+).*/\1.\2.\3.\4/')"
export MASTER_PORT="$(( 20000 + SLURM_JOB_ID % 20000 ))"
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_ASYNC_ERROR_HANDLING=1
_IFACE="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
export NCCL_SOCKET_IFNAME="$_IFACE" GLOO_SOCKET_IFNAME="$_IFACE"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=x ATTN_BACKEND=flash_attn_3 FUSED_MODULATE=1
echo "================ HOLD 2-NODE UP  job=$SLURM_JOB_ID ================"
echo " nodes=$SLURM_NODELIST  master=$MASTER_ADDR:$MASTER_PORT  iface=$_IFACE"
echo " SSH/attach a shell:  srun --jobid=$SLURM_JOB_ID --overlap --nodes=1 --pty bash"
echo " re-run 2-node debug: srun --jobid=$SLURM_JOB_ID --overlap --nodes=2 --ntasks-per-node=1 bash scripts/dbg_2node_inner.sh"
echo "=================================================================="

# (1) auto-run the full-mode + EFA 2-node debug once (25min timeout so a hang doesn't wedge the hold)
echo "[hold] auto-running full-mode+EFA 2-node debug (mini-MDS, 25 steps, 25min cap) ..."
MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
  timeout 1500 srun --overlap --nodes=2 --ntasks-per-node=1 bash scripts/dbg_2node_inner.sh 2>&1 | grep -vE "resampling|load failed"
echo "[hold] auto-debug exited (rc=${PIPESTATUS[0]}). Now HOLDING for interactive re-runs. scancel $SLURM_JOB_ID to release."

# (2) hold the allocation
srun --overlap --ntasks-per-node=1 sleep infinity
