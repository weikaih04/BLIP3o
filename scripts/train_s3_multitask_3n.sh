#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage-3 TRI-MODAL multitask (I1 .5 / IM .3 / T .2) on the NEW (xgen-mm) cluster.
#
# LAYOUT: one STAGE per node, all three in parallel — ss/shape/tex are independent
# during training (tex conditions on GT shape coords, not on the shape model's
# output; the cascade only couples at inference). Mirrors the repo's original
# design, experiment/qwen35_trellis2_training/train_s3_{ss,shape,tex}_8g.yaml.
# Each stage = 1 node x 8 GPUs, bs8 x ga4 = the standard eff BS 256.
#
# Runs INSIDE an already-held single-node job via `srun --jobid=<id> --overlap`
# (no queue wait, no extra node budget, and the hold is never cancelled).
#
# DATA: the per-asset npz caches on /fsx, via --mixture_config. NOT MDS — the new
# cluster's FSx sustains ~135 samples/s at 4 dataloader workers and ~731 at 16,
# against a ~100 samples/s requirement.
#   !! The reason the mixture path looked IO-bound on the old cluster was NOT the
#      filesystem: accelerate re-shards every IterableDataset and discards
#      (N-1)/N of what it loads (8x waste at 8 ranks). Fixed in
#      trellis2_blip3o/data/rank_aware.py — read that file before touching this.
#
# WARM-STARTING FROM AN s3_*_40k CHECKPOINT (not the fusion ckpts)? Those were trained with
#   cond_pos_stamp=False and view_embed_mode=sincos @ VIEW_EMBED_SCALE=0.2. The per-stage ARCH
#   defaults below target the FUSION ckpts, so override:
#     ARCH='--cond_adapter mlp --cond_pos_stamp False --view_embed_mode sincos' VIEW_EMBED_SCALE=0.2
#   Getting this wrong silently re-inits pos_stamp and lets the fixed sincos table drift.
#
# Usage:
#   STAGE=ss    JOB=193 bash scripts/train_s3_multitask_3n.sh
#   STAGE=shape JOB=194 bash scripts/train_s3_multitask_3n.sh
#   STAGE=tex   JOB=202 bash scripts/train_s3_multitask_3n.sh
#   env: MAX_STEPS(3000) LR(1e-4) DINO_DROP(0.3) PER_GPU_BS(8) WORKERS(10)
#        RUN_TAG(s3tri) DEEPSPEED(configs/deepspeed_zero1.json) REPORT_TO(none)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO=/fsx/home/weikai.huang/3dgen/model/BLIP3o
STAGE="${STAGE:?STAGE must be ss|shape|tex}"
JOB="${JOB:?JOB = the held slurm job id to run inside (193|194|202)}"
RUN_TAG="${RUN_TAG:-s3tri}"
MAX_STEPS="${MAX_STEPS:-3000}"
PER_GPU_BS="${PER_GPU_BS:-8}"
WORKERS="${WORKERS:-10}"       # 96 vCPU / 8 GPUs = 12 cores per GPU
DINO_DROP="${DINO_DROP:-0.3}"
LR="${LR:-1e-4}"
LOGD="${LOGD:-$REPO/runs/cache_logs}"

# Per-stage warm start = the single-image (fusion) production checkpoints, plus the
# ARCHITECTURE flags those checkpoints were trained with. Getting these wrong silently
# drops trained tensors (load_state_dict strict=False) — verified against each ckpt's
# config.json + safetensors key list:
#   * every fusion ckpt has cond_adapter=None, which the model treats as the MLP
#     connector; train_native.py's DEFAULT is "xf2" (a ~27M 2-block transformer), so
#     --cond_adapter mlp is REQUIRED or the whole connector is re-initialised.
#   * only the SS ckpt has diffusion_connector.pos_stamp.{dpos,scale}
#     (config cond_pos_stamp=true) → --cond_pos_stamp True on ss ONLY. dpos is a
#     persistent buffer, so the ckpt's own table is restored over whatever
#     scripts/make_dino_pos.py wrote into runs/cache_logs/dino_pos32.npz.
case "$STAGE" in
  ss)    INIT="${INIT:-runs/fusion_ss_dpos_2n/checkpoint-22000}"
         MIX="${MIX:-configs/mix_s3_multitask_xgenmm_ss.yaml}"   # ss_only → no shape/pbr IO
         ARCH="${ARCH:---cond_adapter mlp --cond_pos_stamp True}" ;;
  shape) INIT="${INIT:-runs/fusion_shape_v22gv/checkpoint-16000}"
         MIX="${MIX:-configs/mix_s3_multitask_xgenmm.yaml}"
         ARCH="${ARCH:---cond_adapter mlp}" ;;
  tex)   INIT="${INIT:-runs/fusion_tex_v22gv/checkpoint-16000}"
         MIX="${MIX:-configs/mix_s3_multitask_xgenmm.yaml}"
         ARCH="${ARCH:---cond_adapter mlp}" ;;
  *) echo "STAGE must be ss|shape|tex (got '$STAGE')" >&2; exit 1 ;;
esac
[ -e "$REPO/$INIT" ] || { echo "[s3tri] INIT not found: $REPO/$INIT" >&2; exit 1; }
[ -e "$REPO/$MIX" ]  || { echo "[s3tri] MIX  not found: $REPO/$MIX"  >&2; exit 1; }
NODE=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null)
[ -n "$NODE" ] || { echo "[s3tri] job $JOB is not running" >&2; exit 1; }

mkdir -p "$LOGD"
LOG="$LOGD/s3tri_${STAGE}.log"
echo "[s3tri] STAGE=$STAGE job=$JOB node=$NODE"
echo "[s3tri] mix=$MIX init=$INIT arch='$ARCH'"
echo "[s3tri] bs=$PER_GPU_BS x 8 gpus x ga4 = eff 256, workers=$WORKERS, max_steps=$MAX_STEPS"
echo "[s3tri] out=runs/s3_${STAGE}_${RUN_TAG}  log=$LOG"

setsid nohup srun --jobid="$JOB" --overlap -n1 bash -c '
  set -u
  source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis
  cd '"$REPO"'
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TORCHINDUCTOR_CACHE_DIR=/dev/shm/ind_s3_'"$STAGE"'
  export TRITON_CACHE_DIR=/dev/shm/tri_s3_'"$STAGE"'
  export TMPDIR=/dev/shm/tmpc_s3_'"$STAGE"'
  # COMPILE_SS=0: cond length swings 60 (T) / 292 (IM) / 1054 (I1) tokens per batch,
  # which makes torch.compile recompile the SS flow constantly.
  env PER_GPU_BS='"$PER_GPU_BS"' MAX_STEPS='"$MAX_STEPS"' LR='"$LR"' \
      WORKERS='"$WORKERS"' DINO_DROP='"$DINO_DROP"' RUN_TAG='"$RUN_TAG"' \
      MIX='"$MIX"' REPORT_TO='"${REPORT_TO:-none}"' COMPILE_SS=0 \
      VIEW_EMBED_SCALE='"${VIEW_EMBED_SCALE:-1.0}"' \
      DEEPSPEED='"${DEEPSPEED:-configs/deepspeed_zero1.json}"' \
      EXTRA_ARGS="--save_steps ${SAVE_STEPS:-3000} --save_total_limit ${SAVE_KEEP:-4} '"$ARCH"'" \
      bash scripts/train_native_split.sh '"$STAGE"' s3 '"$INIT"' 8
' > "$LOG" 2>&1 &
echo "[s3tri] launched. tail -f $LOG"
