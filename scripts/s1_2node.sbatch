#!/bin/bash
#SBATCH --job-name=s1_2node
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=2
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/s1_2node-%j.out
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export MASTER_PORT=29511
srun --ntasks=2 --ntasks-per-node=1 bash /fsx/sfr/weikaih/3dgen/model/BLIP3o/scripts/s1_2node_launch.sh
