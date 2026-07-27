#!/bin/bash
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --job-name=vggt2048_tiny
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --time=0-00:30:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/vggt2048_tiny_%j.out
# TINY sanity build for the RAW-2048-d sparse REPA targets: 1 GPU, --limit 32.
set -u
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export HF_HUB_OFFLINE=1
MANI="/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_repa16k.jsonl"
OUT="/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/vggt2048_tiny"
mkdir -p "$OUT" runs/cache_logs
echo "[tiny] manifest=$MANI out=$OUT node=$(hostname)"
CUDA_VISIBLE_DEVICES=0 python scripts/build_vggt_targets.py \
  --manifest "$MANI" --out "$OUT" --shard 0 --num_shards 1 --limit 32
echo "[tiny] DONE nfiles=$(find "$OUT" -name vggt16.npz | wc -l)"
