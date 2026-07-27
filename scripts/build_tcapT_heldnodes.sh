#!/bin/bash
# ── T caption cond cache on ALREADY-HELD nodes (contingency fast path) ───────────
# Same build as scripts/build_tcapT_lowpri.sbatch (sbatch jobs 226/227), but runs the 8
# GPU shards inside an existing sleep-infinity / hold job via `srun --jobid --overlap`,
# so it needs NO queue wait and consumes NO extra node budget. Does NOT scancel anything.
#
# WHY THIS EXISTS: as of 2026-07-24 22:5x all 23 GPU nodes are allocated by tier-20 jobs,
# so the tier-10 low-pri jobs 226/227 are PENDING with a slurm start estimate ~30 h out.
# The whole T build is only ~3.3 GPU-hours (≈25-60 min on one 8-GPU node), so the moment
# a held node has idle GPUs — e.g. after the IM/im4r cache build on jobs 193/194/202
# finishes — this is the cheapest way to land it.
#
# DO NOT RUN THIS WHILE THE IM CACHE BUILD IS STILL USING THOSE GPUS. Check first:
#   srun --jobid=<id> --overlap -n1 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
# and only proceed when the GPUs are idle. Then:
#   JOBS=193 bash scripts/build_tcapT_heldnodes.sh            # one held node, 8 shards
#   JOBS="193 194" bash scripts/build_tcapT_heldnodes.sh      # two nodes, 16 shards
#
# If sbatch jobs 226/227 have meanwhile started, cancel the redundant one rather than
# running both — though it is harmless if you don't: every worker skips t-keys whose npz
# already exists, so overlapping runs only waste a little compute, never corrupt output.
set -uo pipefail

JOBS=${JOBS:-"193"}
REPO=/fsx/home/weikai.huang/3dgen/model/BLIP3o
MD=/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered
MANIS=${MANIS:-$MD/vlm_filtered_capT.jsonl}
OUT=${OUT:-/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1}
CK_SRC=/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000
VLM_NAME=${VLM_NAME:-/dev/shm/v22ckpt}   # preserve the prod root's recorded contract string
LOGD=${LOGD:-$REPO/runs/cache_logs/tcapT_new}

NNODE=$(echo $JOBS | wc -w)
NSHARD=$((NNODE * 8))
mkdir -p "$LOGD"

NODE=0
for JID in $JOBS; do
  setsid nohup srun --jobid="$JID" --overlap -n1 bash -c '
    set -uo pipefail
    source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis
    cd '"$REPO"'
    export ATTN_BACKEND=sdpa TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
    NODE='"$NODE"'; NSHARD='"$NSHARD"'; LOGD='"$LOGD"'
    CK_LOCAL=/dev/shm/v22ckpt_t
    mkdir -p "$CK_LOCAL"
    rsync -a --exclude "rng_state_*.pth" --exclude "global_step*" --exclude "*.optim*" \
      --exclude latest '"$CK_SRC"'/ "$CK_LOCAL"/ 2>&1 | tail -1
    echo "[tcapT] $(hostname) node=$NODE staged ckpt $(date)"
    for g in $(seq 0 7); do
      GS=$((NODE * 8 + g))
      (
        for try in 1 2 3 4 5 6; do
          CUDA_VISIBLE_DEVICES=$g python scripts/build_vlm_cache_v22.py \
            --mode captions --manifests '"$MANIS"' --out_root '"$OUT"' \
            --vlm "$CK_LOCAL" --vlm_name '"$VLM_NAME"' \
            --batch_size 32 --shard "$GS" --num_shards "$NSHARD" \
            >> "$LOGD/t_held_$GS.log" 2>&1
          rc=$?
          [ $rc -eq 0 ] && break
          echo "[tcapT] shard $GS exited rc=$rc, retry $try $(date)" >> "$LOGD/t_held_$GS.log"
          sleep 30
        done
      ) &
    done
    wait
    echo "[tcapT] $(hostname) node=$NODE ALL DONE $(date)"' \
    > "$LOGD/held_node${NODE}_job${JID}.out" 2>&1 &
  echo "launched node $NODE on held job $JID (shards $((NODE*8))..$((NODE*8+7)) of $NSHARD)"
  NODE=$((NODE + 1))
done
echo "OUT=$OUT  LOGD=$LOGD  NSHARD=$NSHARD"
