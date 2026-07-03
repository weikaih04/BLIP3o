#!/bin/bash
# Stage-1 (connector warmup) on ready_v3, 2 nodes / 16×H200, NODE-SHARDED MDS.
# xgen-mm (normal, no preemption; pends on AssocGrpNodeLimit when shared quota full).
# --requeue + auto-resume kept as node-failure insurance: train_native.py auto-resumes from runs/s1_v3/checkpoint-*
# (output_dir on /fsx persists across requeue) + StreamingDataset deterministic resume;
# stage_mds_v3.sh is idempotent (skips shards already on NVMe). 10 epochs of 931,365 @ eff-BS 256.
#SBATCH --job-name=s1v3_2node
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=2
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/s1v3_2node-%j.out
# MASTER as IPv4 (zhiyuan: hostname can socket-timeout on some nodes). EFA NCCL env is in s1_v3_launch.sh.
_head="$(scontrol show hostnames "$SLURM_NODELIST" | head -1)"
export MASTER_ADDR="$(getent ahostsv4 "$_head" 2>/dev/null | awk '{print $1; exit}')"
[ -z "$MASTER_ADDR" ] && export MASTER_ADDR="$(echo "$_head" | sed -E 's/^ip-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+).*/\1.\2.\3.\4/')"
export MASTER_PORT="$(( 20000 + SLURM_JOB_ID % 20000 ))"
# 2 epochs: 931365 / (bs2 × ga8 × 16gpu = 256) ≈ 3638 steps/epoch × 2 = 7276
export MAX_STEPS=7280
export OUTDIR=runs/s1_v3
export PER_GPU_BS=2 GA=8
export MDS_MODE=full          # FOLLOW ZHIYUAN: full MDS per node + standard global sharding (no node-local patch)
srun --ntasks=2 --ntasks-per-node=1 bash /fsx/sfr/weikaih/3dgen/model/BLIP3o/scripts/s1_v3_launch.sh
