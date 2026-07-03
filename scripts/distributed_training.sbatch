#!/bin/bash
# 1-node held allocation (8×H200) on xgen-mm. Holds the node so you can srun/ssh in and run
# training/debug interactively WITHOUT re-queuing. EFA + IPv4-master env pre-exported so a
# multi-node run (if you later add nodes) or a single-node run is ready to go.
# ⚠️ Idle GPUs violate the cluster ≥50%-util rule → run compute promptly, don't leave it sleeping.
#SBATCH --job-name=distributed_training
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=1-00:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/distributed_training-%j.out
export MASTER_ADDR="$(getent ahostsv4 "$(hostname)" 2>/dev/null | awk '{print $1; exit}')"
[ -z "$MASTER_ADDR" ] && export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="$(( 20000 + SLURM_JOB_ID % 20000 ))"
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_ASYNC_ERROR_HANDLING=1
_IFACE="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
export NCCL_SOCKET_IFNAME="$_IFACE" GLOO_SOCKET_IFNAME="$_IFACE"
echo "======== distributed_training node UP  job=$SLURM_JOB_ID  node=$SLURM_NODELIST ========"
echo " attach a shell:  srun --jobid=$SLURM_JOB_ID --overlap --pty bash"
echo " (8×H200 held. run training here promptly — cluster flags <50% GPU util.)"
echo "================================================================================"
srun --ntasks-per-node=1 sleep infinity
