#!/bin/bash
# Single-node S1 (connector warmup) on full ready_v3 (931k), 1 node / 8×H200. The PROVEN path
# (s1_run + probe 16994 both single-node, clean ~8.65s/it). Sidesteps the 2-node NCCL hang.
# Stages all 4 shards onto ONE node (193-194 already has 0+2 → only builds 1+3, ~1h), then trains
# on all 4. bs2 × ga16 × 8 = eff-BS 256 (matches the intended batch). 2 epochs = 7276 steps ≈ ~35h.
# auto-resume from runs/s1_v3_1node/checkpoint-* on requeue.
#SBATCH --job-name=s1v3_1node
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --nodelist=ip-10-0-193-194
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/s1v3_1node-%j.out
export MAX_STEPS=7276            # 931365×2 / (bs2×ga16×8 = 256)
export OUTDIR=runs/s1_v3_1node
export PER_GPU_BS=2 GA=16        # eff-BS 256 on 8 GPU
srun --ntasks=1 --ntasks-per-node=1 bash /fsx/sfr/weikaih/3dgen/model/BLIP3o/scripts/s1_v3_launch.sh
