#!/bin/bash
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --job-name=vggt2048_full
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --time=0-02:00:00
#SBATCH --output=/fsx/sfr/weikaih/3dgen/model/BLIP3o/runs/cache_logs/vggt2048_full_%j.out
# Full 16k RAW-2048-d sparse REPA target cache, 8-way GPU shard (~15 min compute).
set -u
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
export HF_HUB_OFFLINE=1
MANI="/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_repa16k.jsonl"
OUT="/fsx/home/weikai.huang/3dgen/im_probe/vggt2048_targets"
mkdir -p "$OUT" runs/cache_logs
echo "[full2048] manifest=$MANI out=$OUT node=$(hostname)"
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g python scripts/build_vggt_targets.py \
    --manifest "$MANI" --out "$OUT" --shard "$g" --num_shards 8 \
    > "runs/cache_logs/vggt2048_full_w$g.log" 2>&1 &
done
wait
echo "[full2048] DONE nfiles=$(find "$OUT" -name vggt16.npz | wc -l)"
