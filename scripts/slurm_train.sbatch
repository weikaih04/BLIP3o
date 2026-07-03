#!/bin/bash
#SBATCH --job-name=blip3o_native
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%x-%j.out
# ─────────────────────────────────────────────────────────────────────────────
# Single-node SLURM wrapper for the native-VLM training launchers.
# WE USE SLURM, NOT BEAKER — the experiment/qwen35_trellis2_training/*.yaml are
# kept only as the ground-truth FLAG reference; this is the actual runner.
#
# The launchers call `torchrun --nproc_per_node=N` internally, so SLURM only has
# to allocate ONE node with N GPUs. Keep --gres=gpu:N == the launcher's NPROC arg.
#
# Usage (the 3-link curriculum — see docs/TRAINING_HANDOFF.md §2):
#   sbatch scripts/slurm_train.sbatch bash scripts/train_connector_warmup.sh 8
#
#   S1=runs/p1_connector_warmup/checkpoint-3000
#   sbatch scripts/slurm_train.sbatch bash scripts/train_native_split.sh ss    fusion $S1 8
#   sbatch scripts/slurm_train.sbatch bash scripts/train_native_split.sh shape fusion $S1 8
#   sbatch scripts/slurm_train.sbatch bash scripts/train_native_split.sh tex   fusion $S1 8
#
#   sbatch scripts/slurm_train.sbatch bash scripts/train_native_split.sh ss    s3 runs/fusion_ss/checkpoint-3000    8
#   sbatch scripts/slurm_train.sbatch bash scripts/train_native_split.sh shape s3 runs/fusion_shape/checkpoint-3000 8
#   sbatch scripts/slurm_train.sbatch bash scripts/train_native_split.sh tex   s3 runs/fusion_tex/checkpoint-3000   8
#
# Override resources per-submit, e.g. (ss is light → 4 GPU; launcher rebalances grad_accum):
#   sbatch --gres=gpu:4 scripts/slurm_train.sbatch bash scripts/train_native_split.sh ss fusion $S1 4
#
# Multi-node (the eff-BS-256 chain is single-node 8-GPU, so not needed) would require
# srun + torchrun --rdzv_* — see scripts/run.sh for that header.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Env (mirror the perf-critical bits of the Beaker jobs).
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=flash_attn
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=trellis2_blip3o
# WANDB_API_KEY only required when logging to wandb (launchers honor REPORT_TO; none = key-less).
# Set HF_HOME to a persistent cache if Qwen3.5-2B isn't already downloaded.
if [ "${REPORT_TO:-wandb}" != "none" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[slurm_train] REPORT_TO=${REPORT_TO:-wandb} but WANDB_API_KEY is unset." >&2
  echo "              Export WANDB_API_KEY, or submit with REPORT_TO=none for a key-less run." >&2
  exit 1
fi

echo "[slurm_train] node=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-?} job=${SLURM_JOB_ID:-none}"
echo "[slurm_train] cmd: $*"
exec "$@"
