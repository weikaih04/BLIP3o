#!/bin/bash
# ── IM (multi-image) 4-view VLM cond cache, WEIGHTED-RANDOM views → v22_im4r ──────
# NEW cluster (ip-10-1-x). Runs 8 GPU shards on each of the already-held sleep-infinity
# jobs via `srun --jobid=<id> --overlap` (no queue wait; does NOT scancel the holds).
#
# vs the old scripts/build_im4_2n.sbatch (old cluster): paths /fsx/sfr/weikaih →
# /fsx/home/weikai.huang, 2 nodes → 3, and view picking is --im4_view_sampling weighted
# (4 distinct of ALL 16 renders, p ∝ elevation-band weights, per-sha seeded) instead of
# the fixed 005-011 good-band offsets. Everything else — qwen-light 64 tok/view (→292
# joint tokens), DINOv3 @320 (405 tok/view ×4 = 1620), m00 key, npz field set — is
# byte-format identical to v22_im4l so threed.py / streaming_task.py consume it unchanged.
#
# Idempotent: skips any m00.npz already ≥1 MB, and the per-sha RNG reproduces the same
# 4 views, so preemption / restarts / partial reruns converge to the same cache.
#
#   bash scripts/build_im4r_heldnodes.sh            # uses JOBS below
#   JOBS="193 194 202" OUT=... bash scripts/build_im4r_heldnodes.sh
set -uo pipefail

JOBS=${JOBS:-"193 194 202"}                       # held sleep-infinity job ids
OUT=${OUT:-/fsx/home/weikai.huang/3dgen/vlm_hidden_cache/v22_im4r}
MANI=${MANI:-/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_all.jsonl}
REPO=/fsx/home/weikai.huang/3dgen/model/BLIP3o
LOGD=${LOGD:-$REPO/runs/cache_logs/im4r}
# QWEN-LIGHT budget (identical to the v22_im4l full-scale run): qwen 64 tok/view = the
# processor FLOOR (256² input, patch16/merge2 → 8×8=64); 4-img joint → 292 qwen tok.
# DINO @320 → 405 tok/view (1620 for 4). ≈4.53 MB/entry, 420k ≈ 1.90 TB.
IM_TOK=${IM_TOK:-64}
DINO_SZ=${DINO_SZ:-320}
MIN_FREE_TB=${MIN_FREE_TB:-2.0}
CK_SRC=/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000

NNODE=$(echo $JOBS | wc -w)
NSHARD=$((NNODE * 8))
mkdir -p "$LOGD" "$OUT"

NODE=0
for JID in $JOBS; do
  # detached (setsid) so the srun step outlives this shell / the launching session
  setsid nohup srun --jobid="$JID" --overlap -n1 bash -c '
    set -uo pipefail
    source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis
    cd '"$REPO"'
    export ATTN_BACKEND=sdpa TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
    NODE='"$NODE"'; NSHARD='"$NSHARD"'; LOGD='"$LOGD"'
    CK_LOCAL=/dev/shm/v22ckpt_im4r
    mkdir -p "$CK_LOCAL"
    # stage the 2B ckpt to node RAM (weights only) — 8 procs hammering /fsx is wasteful
    rsync -a --exclude "rng_state_*.pth" --exclude "global_step*" --exclude "*.optim*" \
      --exclude latest '"$CK_SRC"'/ "$CK_LOCAL"/ 2>&1 | tail -1
    echo "[im4r] $(hostname) node=$NODE staged ckpt $(date)"
    for g in $(seq 0 7); do
      GS=$((NODE * 8 + g))
      (
        # retry loop: the worker is idempotent, so a transient crash just re-scans and
        # continues. exit 17 = the disk guard fired → FATAL, never retry.
        for try in 1 2 3 4 5; do
          CUDA_VISIBLE_DEVICES=$g python scripts/build_vlm_cache_v22.py \
            --mode im4 --im4_view_sampling weighted \
            --manifests '"$MANI"' --out_root '"$OUT"' \
            --vlm "$CK_LOCAL" --vlm_name '"$CK_SRC"' \
            --im_tok_per_view '"$IM_TOK"' --dino_image_size '"$DINO_SZ"' \
            --min_free_tb '"$MIN_FREE_TB"' \
            --shard "$GS" --num_shards "$NSHARD" >> "$LOGD/im_$GS.log" 2>&1
          rc=$?
          [ $rc -eq 0 ] && break
          [ $rc -eq 17 ] && { echo "[im4r] shard $GS DISK-GUARD ABORT" >> "$LOGD/im_$GS.log"; break; }
          echo "[im4r] shard $GS exited rc=$rc, retry $try $(date)" >> "$LOGD/im_$GS.log"
          sleep 30
        done
      ) &
    done
    wait
    echo "[im4r] $(hostname) node=$NODE ALL DONE $(date)"' \
    > "$LOGD/node${NODE}_job${JID}.out" 2>&1 &
  echo "launched node $NODE on job $JID  (shards $((NODE*8))..$((NODE*8+7)) of $NSHARD)"
  NODE=$((NODE + 1))
done
echo "OUT=$OUT  LOGD=$LOGD  NSHARD=$NSHARD"
