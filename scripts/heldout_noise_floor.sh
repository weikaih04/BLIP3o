#!/bin/bash
# Measure the NOISE FLOOR of the clean held-out set.
#
# A held-out set is only as good as the effect size it can resolve.  We run the SAME probe
# over the SAME assets under conditions that should NOT change the answer:
#   * two nearby converged checkpoints of ONE run (s3_ss_40k 28000 / 30000 — 2000 steps
#     apart at lr ~1.9e-5, i.e. genuinely similar models), and
#   * two sampler seeds.
# Anything that moves is noise.  4 runs = a 2x2, which separates checkpoint noise from
# sampler noise and gives a per-asset sd so unstable assets can be named, not averaged in.
#
#   GPUS=0,1,2,3 bash scripts/heldout_noise_floor.sh
#   GPUS=0,1,2,3 MANI=.../heldout14_newpaths.jsonl COND_IM=.../v22_heldout_im4r \
#       TAGPFX=old14 bash scripts/heldout_noise_floor.sh     # the n=14 comparison run
set -uo pipefail
REPO=/fsx/home/weikai.huang/3dgen/model/BLIP3o
source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis
cd "$REPO"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/fsx/home/weikai.huang/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}   # shared node: don't let N workers each
export MKL_NUM_THREADS=$OMP_NUM_THREADS        # grab all 96 cores during model load

HO=/fsx/home/weikai.huang/3dgen/im_probe/heldout_clean
MANI=${MANI:-$HO/heldout_clean.jsonl}
COND_IM=${COND_IM:-/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_clean_im4r}
CKPTS=${CKPTS:-"runs/s3_ss_40k/checkpoint-28000 runs/s3_ss_40k/checkpoint-30000"}
SEEDS=${SEEDS:-"0 1"}
TAGPFX=${TAGPFX:-clean}
# MODE=diag -> diag_im_multiview.py   (4-distinct vs 4-copy vs 1-view, one model)
# MODE=eval -> eval_im_vs_i1_ss.py    (IM 4-view vs the deployed I1 baseline; needs COND_I1)
MODE=${MODE:-diag}
case "$MODE" in
  diag) PROBE=scripts/diag_im_multiview.py; PFX=diag_im_mv ;;
  eval) PROBE=scripts/eval_im_vs_i1_ss.py; PFX=im_probe_iou ;;
  *) echo "MODE must be diag|eval"; exit 1 ;;
esac
COND_I1=${COND_I1:-/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_heldout}
export COND_I1
export N_GRID=${N_GRID:-0}
GPUS=${GPUS:-0,1,2,3}
IFS=',' read -ra GARR <<< "$GPUS"
NG=${#GARR[@]}
LOGD=$REPO/runs/cache_logs/heldout_noise; mkdir -p "$LOGD"
echo "[noise] mani=$MANI n=$(wc -l < "$MANI") gpus=$GPUS ckpts=$CKPTS seeds=$SEEDS"

for CK in $CKPTS; do
  for SD in $SEEDS; do
    TAG="${TAGPFX}_$(basename "$CK")_s${SD}"
    echo "[noise] === $TAG === $(date)"
    i=0
    for g in "${GARR[@]}"; do
      CUDA_VISIBLE_DEVICES=$g SS_CKPT_IM=$REPO/$CK COND_IM=$COND_IM MANI=$MANI \
        SEED=$SD SHARD=$i NUM_SHARDS=$NG OUT_TAG="${TAG}_sh${i}of${NG}" \
        python $PROBE > "$LOGD/${TAG}_$i.log" 2>&1 &
      i=$((i+1))
    done
    wait
    grep -hE "DIAG_DONE|IM_EVAL_DONE" "$LOGD/${TAG}"_*.log | wc -l \
      | xargs echo "[noise]   shards done:"
    grep -hE "Traceback|Error" "$LOGD/${TAG}"_*.log | head -3
  done
done

echo "[noise] ===== REDUCE ====="
python scripts/reduce_diag_im.py \
  --glob "runs/cache_logs/${PFX}_${TAGPFX}_*.json" \
  --manifest "$MANI" --out "$HO/noise_floor_${MODE}_${TAGPFX}.json"
