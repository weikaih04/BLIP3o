#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ②③ Per-flow SPLIT trainer (ss / shape / tex as separate jobs).
# Ground truth: experiment/qwen35_trellis2_training/train_{fusion,s3}_{ss,shape,tex}_8g.yaml
# (commit 96a0b26). Each job = its own connector copy + ONE flow, init-warm-started
# from the previous link, GT-decoupled. Assembled at inference.
#
# Curriculum chain:
#   ① scripts/train_connector_warmup.sh            → runs/p1_connector_warmup/checkpoint-3000
#   ② this, ROUND=fusion, INIT=①                    → runs/fusion_<stage>/checkpoint-3000   (I1 cached+fusion)
#   ③ this, ROUND=s3,     INIT=②fusion_<stage>      → runs/s3_<stage>/checkpoint-3000       (multitask I1+IM+T)
#
# H200 overrides vs ground truth (per weikaih 2026-06-25):
#   • elastic_target_ratio 0.1 → 0.75   (H200 141G headroom; shape/tex only)
#   • dino_drop_prob → 0.3 BOTH rounds  (latest; ground-truth fusion was 0.1, s3 was 0.3)
#
# Usage: bash scripts/train_native_split.sh STAGE ROUND INIT_CKPT [NPROC=8]
#   STAGE ∈ {ss, shape, tex}
#   ROUND ∈ {fusion, s3}
#   INIT_CKPT = checkpoint dir to warm-start from (REQUIRED — fusion←S1, s3←fusion_<stage>)
#   env: PER_GPU_BS(8) ELASTIC_RATIO(0.75) DINO_DROP(0.3) MAX_STEPS(3000) LR(1e-4) RUN_TAG
#   Holds eff BS 256: ga = 256 / (PER_GPU_BS × NPROC) → bs8×nproc8 = ga4.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
STAGE="${1:?usage: train_native_split.sh STAGE(ss|shape|tex) ROUND(fusion|s3) INIT_CKPT [NPROC]}"
ROUND="${2:?ROUND must be fusion|s3}"
INIT="${3:?INIT_CKPT (checkpoint dir to warm-start from) is required}"
NPROC="${4:-8}"
PER_GPU_BS="${PER_GPU_BS:-8}"   # bs8 = the lossless ~30% MFU sweet spot (≈105GB/gpu @ 8-GPU, at the
ELASTIC_RATIO="${ELASTIC_RATIO:-0.75}"
DINO_DROP="${DINO_DROP:-0.3}"
MAX_STEPS="${MAX_STEPS:-3000}"
LR="${LR:-1e-4}"
# NOTE: /fsx/sfr/weikaih is the OLD cluster; the new (xgen-mm) cluster's env lives under
# /fsx/home/weikai.huang. Try the new path first, keep the old as a fallback.
if ! command -v torchrun >/dev/null 2>&1; then
  source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis 2>/dev/null \
    || source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
fi
cd "$(dirname "$0")/.."
export PATH="$(dirname "$(command -v python)"):$PATH"
# torch.compile cache — MUST be LOCAL. Pointing TRITON_CACHE_DIR at /fsx (Lustre) crashes with
# "OSError [Errno 14] Bad address" (triton mmaps its .llir/.so; Lustre doesn't support it).
# So cache lives on /dev/shm (per node). Persistence across grabs is done by rsync
# (/dev/shm build → save to /fsx → restore to /dev/shm next time), NOT by cache-dir-on-Lustre.
# Respect caller-provided cache dirs (some nodes have /dev/shm unwritable → caller points these
# at node-local NVMe); only default to /dev/shm when unset.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/ind_${SLURM_NODEID:-0}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/dev/shm/tri_${SLURM_NODEID:-0}}"
export TMPDIR="${TMPDIR:-/dev/shm/tmpc_${SLURM_NODEID:-0}}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR" 2>/dev/null || true

# Effective batch size. Default 256 (the recipe every prior link in the chain used).
# Overridable because 256 is NOT reachable on an ODD node count: 3 nodes × 8 GPUs = 24
# ranks and 24 ∤ 256 (256 = 2^8 has no factor 3). The 3-node S3 multitask run therefore
# uses EFF_BS=384 (bs8 × 24 ranks × ga2). Keep it a power-of-two multiple of the world
# size or this assert fires.
EFF_BS="${EFF_BS:-256}"
NNODES="${NNODES:-1}"
WORLD=$(( PER_GPU_BS * NPROC * NNODES ))
GA=$(( EFF_BS / WORLD ))
[ $(( WORLD * GA )) -eq $EFF_BS ] || {
  echo "[split] PER_GPU_BS($PER_GPU_BS) × NPROC($NPROC) × NNODES($NNODES) must divide EFF_BS($EFF_BS)" >&2; exit 1; }
# multi-node: torchrun rendezvous + zhiyuan NCCL/EFA env (harmless single-node; the
# bs8 shape kills the variable-length straggler — see memory train-native-multinode-straggler)
DIST_FLAGS=""
if [ "$NNODES" -gt 1 ]; then
  DIST_FLAGS="--nnodes=$NNODES --node-rank=${NODE_RANK:?NODE_RANK required for NNODES>1} \
              --master-addr=${MASTER_ADDR:?} --master-port=${MASTER_PORT:-29601}"
  export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
  export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_ASYNC_ERROR_HANDLING=1
  _IF="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
  export NCCL_SOCKET_IFNAME="$_IF" GLOO_SOCKET_IFNAME="$_IF"
fi
# INIT="none" → skip warm-start (speed profiling: init weights don't affect step time).
if [ "$INIT" = "none" ]; then INIT_FLAG="";
else [ -e "$INIT" ] || echo "[split][warn] INIT_CKPT not found on disk: $INIT" >&2; INIT_FLAG="--init_from_checkpoint $INIT"; fi
# DEEPSPEED env → swap ZeRO stage / disable (none = plain DDP).
# Default ZeRO-1: measured ~8% faster than ZeRO-2 for the split (258M trainable → optimizer
# states are tiny, gradient partitioning is pure overhead). zero0 ≈ zero1; zero2 was the GT default.
DS_CFG="${DEEPSPEED:-configs/deepspeed_zero1.json}"
if [ "$DS_CFG" = "none" ]; then
  # plain DDP (no deepspeed engine). Disable find_unused_parameters (no unused params →
  # the default-True graph traversal of the whole flow is pure overhead).
  DS_FLAG="--ddp_find_unused_parameters False"
else
  DS_FLAG="--deepspeed $DS_CFG"
fi
REPORT="${REPORT_TO:-wandb}"

# per-stage flow/build wiring (identical to ground truth). ss has no SLAT → no elastic.
# COMPILE_SS (default 1=True) → --compile_ss_flow. Set COMPILE_SS=0 for IM runs: variable
# cond length thrashes torch.compile (recompiles per new shape) → disable it there.
COMPILE_SS="${COMPILE_SS:-1}"
if [ "$COMPILE_SS" = "0" ]; then SS_COMPILE_FLAG="--compile_ss_flow False";
else SS_COMPILE_FLAG="--compile_ss_flow True --compile_mode default"; fi
case "$STAGE" in
  ss)        SLAT_FLAGS="--build_slat False --ss_only True  $SS_COMPILE_FLAG" ;;
  shape|tex) SLAT_FLAGS="--build_slat True  --ss_only False --compile_ss_flow False --elastic_slat True --elastic_target_ratio ${ELASTIC_RATIO}" ;;
  *) echo "[split] STAGE must be ss|shape|tex (got '$STAGE')" >&2; exit 1 ;;
esac

# per-round data/cond wiring. s3 raises cond_max_length for 4-view fusion (~8225 tok).
case "$ROUND" in
  fusion) MIX="${MIX:-configs/mix_i1_cached_fusion_local.yaml}"; CONDMAX="";                      WORKERS="${WORKERS:-4}" ;;
  s3)     MIX="${MIX:-configs/mix_s3_multitask.yaml}"; CONDMAX="--cond_max_length 10240"; WORKERS="${WORKERS:-8}" ;;
  *) echo "[split] ROUND must be fusion|s3 (got '$ROUND')" >&2; exit 1 ;;
esac

# Data source: MDS_ROOT (MosaicML Streaming, node/rank-sharded + resumable) overrides the
# manifest mixture. Accepted for fusion (single-task I1 cond) AND s3 (single-task IM cond) —
# both pack ONE cond stream per sample, so a single MDS dataset covers them. (True multitask
# mixtures would need one MDS per task; s3-over-MDS here is the IM-only cond path.)
if { [ "$ROUND" = "fusion" ] || [ "$ROUND" = "s3" ]; } && [ -n "${MDS_SHARDS:-}" ]; then
  # node-local MDS shard dirs (comma-separated) — NVMe-fast; the mixture path starves GPUs
  # when several nodes hammer /fsx with ~15MB/sample random reads (measured 0% GPU util).
  DATA_FLAG="--mds_shards $MDS_SHARDS${MDS_CACHE_LIMIT:+ --mds_cache_limit $MDS_CACHE_LIMIT}"
  DATA_DESC="mds_shards=$MDS_SHARDS"; WORKERS="${MDS_WORKERS:-0}"
elif { [ "$ROUND" = "fusion" ] || [ "$ROUND" = "s3" ]; } && [ -n "${MDS_ROOT:-}" ]; then
  DATA_FLAG="--mds_root $MDS_ROOT${MDS_CACHE_LIMIT:+ --mds_cache_limit $MDS_CACHE_LIMIT}"
  DATA_DESC="mds=$MDS_ROOT"; WORKERS=0          # StreamingDataset shards internally; workers=0
else
  DATA_FLAG="--mixture_config $MIX"; DATA_DESC="mix=$MIX"
fi
OUT="runs/${ROUND}_${STAGE}${RUN_TAG:+_$RUN_TAG}"
echo "[split] stage=$STAGE round=$ROUND nproc=$NPROC bs=$PER_GPU_BS ga=$GA (eff $EFF_BS) "
echo "        dino_drop=$DINO_DROP elastic=$ELASTIC_RATIO $DATA_DESC init=$INIT out=$OUT"

# Lossless speedups (all verified, true-MFU ~10%→~30%): FA3 attention (built from hopper/, ~1.10×;
# override ATTN_BACKEND=flash_attn if FA3 missing on a node) + fused SLAT kernels (FUSED_MODULATE=1,
# default on) + SS torch.compile (--compile_ss_flow True, already set for ss stage above).
export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn_3}"
export FUSED_MODULATE="${FUSED_MODULATE:-1}"

torchrun --nproc_per_node="$NPROC" ${DIST_FLAGS} train_native.py \
  --vlm_model Qwen/Qwen3.5-2B \
  ${DATA_FLAG} \
  --build_vlm False \
  --fuse_dino True --dino_drop_prob "$DINO_DROP" \
  ${CONDMAX} \
  --train_stages "$STAGE" ${SLAT_FLAGS} \
  --freeze_vlm True --flow_tune last20 \
  ${INIT_FLAG} \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --output_dir "$OUT" \
  --max_steps "$MAX_STEPS" --bf16 True \
  --per_device_train_batch_size "$PER_GPU_BS" --gradient_accumulation_steps "$GA" \
  --learning_rate "$LR" --warmup_steps 100 --weight_decay 0.01 \
  --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 \
  --ema_decay 0.998 \
  --logging_steps 5 --save_steps 500 --save_total_limit 4 \
  --report_to "$REPORT" ${DS_FLAG} \
  --ignore_data_skip True --dataloader_num_workers "$WORKERS" ${EXTRA_ARGS:-}
