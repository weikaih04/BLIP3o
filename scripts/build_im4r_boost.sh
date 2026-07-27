#!/bin/bash
# THROUGHPUT BOOST for an in-flight v22_im4r build — adds N extra worker processes per
# GPU WITHOUT touching the running ones (no restart, no lost work, no scancel).
#
# Why: the im4 path is bs=1 and does PIL decode + alpha-crop of 4 webps synchronously in
# the same loop as the forward → measured 16% GPU util, 8.1/141 GB, node load 9.5/96 cores.
# It is CPU/IO-stalled, not GPU-bound, so simply oversubscribing each GPU with more
# processes overlaps one shard's decode with another's forward.
#
# How it stays correct: the builder is idempotent (skips m00.npz ≥1 MB) and the view pick
# is a pure per-sha function, so ANY worker set produces identical bytes. The extra
# workers use a FINER split (--num_shards 96, shards 24..95) and --reverse, so they start
# at the TAIL of their slice — the region the in-flight forward workers have not reached.
# The original 24 forward workers still cover 100% of the manifest on their own; these are
# pure accelerators that converge with them in the middle.
#
#   bash scripts/build_im4r_boost.sh          # 3 extra procs/GPU (→ 4 total)
#   PER_GPU=2 bash scripts/build_im4r_boost.sh
set -uo pipefail

JOBS=${JOBS:-"193 194 202"}
PER_GPU=${PER_GPU:-3}                              # EXTRA procs per GPU (on top of the 1 running)
OUT=${OUT:-/fsx/home/weikai.huang/3dgen/vlm_hidden_cache/v22_im4r}
MANI=${MANI:-/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_all.jsonl}
REPO=/fsx/home/weikai.huang/3dgen/model/BLIP3o
LOGD=${LOGD:-$REPO/runs/cache_logs/im4r}
IM_TOK=${IM_TOK:-64}
DINO_SZ=${DINO_SZ:-320}
MIN_FREE_TB=${MIN_FREE_TB:-2.0}
CK_SRC=/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000

NNODE=$(echo $JOBS | wc -w)
BASE=$((NNODE * 8))                                # 24 = the running forward split
NSHARD=$((BASE * (PER_GPU + 1)))                   # 96 with PER_GPU=3
mkdir -p "$LOGD"

NODE=0
for JID in $JOBS; do
  setsid nohup srun --jobid="$JID" --overlap -n1 bash -c '
    set -uo pipefail
    source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis
    cd '"$REPO"'
    export ATTN_BACKEND=sdpa TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
    NODE='"$NODE"'; BASE='"$BASE"'; NSHARD='"$NSHARD"'; PER_GPU='"$PER_GPU"'; LOGD='"$LOGD"'
    CK_LOCAL=/dev/shm/v22ckpt_im4r                 # already staged by the main launcher
    echo "[boost] $(hostname) node=$NODE +${PER_GPU}/gpu nshard=$NSHARD $(date)"
    for g in $(seq 0 7); do
      SLOT=$((NODE * 8 + g))
      for k in $(seq 1 $PER_GPU); do
        GS=$((k * BASE + SLOT))
        (
          for try in 1 2 3; do
            CUDA_VISIBLE_DEVICES=$g python scripts/build_vlm_cache_v22.py \
              --mode im4 --im4_view_sampling weighted --reverse \
              --manifests '"$MANI"' --out_root '"$OUT"' \
              --vlm "$CK_LOCAL" --vlm_name '"$CK_SRC"' \
              --im_tok_per_view '"$IM_TOK"' --dino_image_size '"$DINO_SZ"' \
              --min_free_tb '"$MIN_FREE_TB"' \
              --shard "$GS" --num_shards "$NSHARD" >> "$LOGD/boost_$GS.log" 2>&1
            rc=$?
            [ $rc -eq 0 ] && break
            [ $rc -eq 17 ] && { echo "[boost] $GS DISK-GUARD ABORT" >> "$LOGD/boost_$GS.log"; break; }
            echo "[boost] $GS rc=$rc retry $try" >> "$LOGD/boost_$GS.log"; sleep 20
          done
        ) &
      done
    done
    wait
    echo "[boost] $(hostname) node=$NODE DONE $(date)"' \
    > "$LOGD/boost_node${NODE}_job${JID}.out" 2>&1 &
  echo "boost: node $NODE on job $JID  +$((PER_GPU*8)) procs (shards of $NSHARD, reverse)"
  NODE=$((NODE + 1))
done
echo "OUT=$OUT LOGD=$LOGD  extra_procs=$((PER_GPU*8*NNODE))"
