#!/bin/bash
# ── T (text_to_3d) caption cond cache → the SHARED S3 cache root ─────────────────
# NEW cluster (ip-10-1-x, /opt/sagemaker scratch, low-pri partitions). Rebuild of the
# t*.npz entries that existed on the old cluster but were NOT migrated: the prod root
# /fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1 currently
# holds only v000 (I1 qwen cond) + d000 (DINO) — no t-keys — so the S3 mixture's
# text_to_3d task (w=0.2) cannot read vlm_cache.caption_key(idx).
#
# vs the old-cluster scripts/build_tcapT_2n.sbatch:
#   * out_root: the SHARED root (S3 mixture points all 3 tasks at ONE cached_hidden_root)
#     instead of a private v22_tcap4 root.
#   * paths /fsx/sfr/weikaih → /fsx/home/weikai.huang.
#   * account/partition xgen-mm/main → low-pri/-low with --requeue (the 23 normal nodes
#     stay with training + the IM cache build).
#   * per-shard retry loop, so a preempt/requeue or a transient crash just re-scans and
#     resumes (the builder skips any t-key whose npz already exists).
# Writes ONLY t000..t003 into the existing {sha[:2]}/{sha}/ dirs — v000/d000 are never
# opened (verified byte-identical md5+mtime after the 32-asset validation run).
#
#   sbatch scripts/build_tcapT_lowpri.sbatch            # forward over the manifest
#   sbatch scripts/build_tcapT_lowpri.sbatch reverse    # 2nd node, tail→head; the two
#                                                       # converge in the middle, and
#                                                       # EITHER one alone still completes
#SBATCH --job-name=weikaih_tcapT
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --output=/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/tcapT_new/slurm_%j.out
set -uo pipefail

REPO=/fsx/home/weikai.huang/3dgen/model/BLIP3o
MD=/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered
MANIS=${MANIS:-$MD/vlm_filtered_capT.jsonl}
OUT=${OUT:-/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1}
CK_SRC=/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000
# The prod root's _meta.json already records vlm="/dev/shm/v22ckpt" (the old build staged
# the same checkpoint-1000 there). captions-mode preserves an EXISTING meta's contract
# fields, so this only matters if _meta.json ever went missing — pin it to the same string
# so a regenerated meta cannot silently break check_meta() for the running I1 jobs.
VLM_NAME=${VLM_NAME:-/dev/shm/v22ckpt}
LOGD=$REPO/runs/cache_logs/tcapT_new
REV=""; [ "${1:-}" = "reverse" ] && REV="--reverse"
TAG="${1:-fwd}"
mkdir -p "$LOGD"

# Stagger the tail-side job: shard 0 of captions-mode rewrites _meta.json (read-modify-
# write, not atomic). Two jobs starting in the same second could have one read it mid-
# truncate. 90 s of separation makes that impossible; content is identical either way.
[ -n "$REV" ] && sleep 90

srun --ntasks-per-node=1 bash -c '
  set -uo pipefail
  source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis
  cd '"$REPO"'
  export ATTN_BACKEND=sdpa TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
  NSHARD=8                      # every job covers the WHOLE manifest with 8 GPU shards
  LOGD='"$LOGD"'; TAG='"$TAG"'
  CK_LOCAL=/dev/shm/v22ckpt_t
  mkdir -p "$CK_LOCAL"
  # stage the 2B ckpt into node RAM — 8 procs pulling 5.3 GB each off Lustre is wasteful
  rsync -a --exclude "rng_state_*.pth" --exclude "global_step*" --exclude "*.optim*" \
    --exclude latest '"$CK_SRC"'/ "$CK_LOCAL"/ 2>&1 | tail -1
  echo "[tcapT] $(hostname) tag=$TAG staged ckpt $(date)"
  for g in $(seq 0 7); do
    (
      # idempotent → a crash/preempt just re-scans and continues from where it stopped
      for try in 1 2 3 4 5 6; do
        CUDA_VISIBLE_DEVICES=$g python scripts/build_vlm_cache_v22.py \
          --mode captions --manifests '"$MANIS"' --out_root '"$OUT"' \
          --vlm "$CK_LOCAL" --vlm_name '"$VLM_NAME"' \
          --batch_size 32 --shard "$g" --num_shards "$NSHARD" '"$REV"' \
          >> "$LOGD/t_${TAG}_$g.log" 2>&1
        rc=$?
        [ $rc -eq 0 ] && break
        echo "[tcapT] shard $g exited rc=$rc, retry $try $(date)" >> "$LOGD/t_${TAG}_$g.log"
        sleep 30
      done
    ) &
  done
  wait
  echo "[tcapT] $(hostname) tag=$TAG ALL DONE $(date)"'
echo "[tcapT] job complete $(date)"
