#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED GEO-TEX DiT — Stage-1 trainer (train_stages=geotex).
# Design: docs/UNIFIED_GEOTEX_DIT_DESIGN.md · gate G0 PASSED 2026-08-11
# ([SCALE]=0.168 → coupling=union default; geo bit-exact, gated tex bit-exact).
#
# What trains: FULL tex stream (1292M) + t-mixer/cross_alpha (+c_gates if gated)
#              + the tex connector. Geo stream + geo connector FROZEN (loaded with
#              EMA overlay from the shape run; enforced by the freeze audit).
# Warm start:  TWO ckpts (not --init_from_checkpoint):
#              shape = runs/s3_shape_t50b/checkpoint-8000
#              tex   = runs/s3_tex_t50b/checkpoint-8000
# Data:        the SAME tri-modal s3 mixture the specialists trained on
#              (mix_s3_multitask_xgenmm.yaml; every sample carries shape+pbr GT).
# Arch flags:  MUST match the t50b ckpts — cond_adapter=mlp, cond_pos_stamp=False,
#              view_embed_mode=sincos @ VIEW_EMBED_SCALE=0.2 (train_native.py also
#              restores the exact table from the tex ckpt as a belt-and-braces).
#
# Usage: bash scripts/train_native_geotex.sh [NPROC=8]
#   env: PER_GPU_BS(4) MAX_STEPS(3000) LR(5e-5) RUN_TAG(v1) WORKERS(10)
#        COUPLING(union) P_CORNER(0.2) P_CORNER2(0.3) DINO_DROP(0.0) EMA(0.9999)
#        MIX(configs/mix_s3_multitask_xgenmm.yaml) REPORT_TO(wandb)
#        DEEPSPEED(configs/deepspeed_zero1.json) EFF_BS(256) EXTRA_ARGS
#   Long runs go through sbatch (scripts/geotex_s1.sbatch) — srun-on-a-hold gets
#   reaped with the hold (memory: long-jobs-need-sbatch).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
NPROC="${1:-8}"
# bs4: the tex stream carries a live geo replica (2× blocks/step vs the tex
# specialist) — start at half the specialist's bs8 and raise after the sanity run.
PER_GPU_BS="${PER_GPU_BS:-4}"
MAX_STEPS="${MAX_STEPS:-3000}"
LR="${LR:-5e-5}"
RUN_TAG="${RUN_TAG:-v1}"
WORKERS="${WORKERS:-10}"
COUPLING="${COUPLING:-union}"
COND_MODE="${COND_MODE:-cross_attn}"   # cross_attn (variant A, S1 default) | stream (variant B A/B arm)
# three-stream MMDiT: cond stream on the LAST N blocks only (+348M at N=10).
COND_STREAM_BLOCKS="${COND_STREAM_BLOCKS:-10}"   # <=0 = all 30 (required when annealing cross-attn away)
XATTN_ANNEAL_START="${XATTN_ANNEAL_START:-0}"
XATTN_ANNEAL_END="${XATTN_ANNEAL_END:-0}"
# t_s=0 corner: geometry clean, texture generated = the tex|mesh mode, which is
# also what the joint sampler's refine pass runs. The geo velocity loss IS masked
# there (flow_heads: the target -x_0 is the input negated, and no inference mode
# reads geo velocity at t_s=0), so this corner is texture budget only.
#
# NOT symmetric with the t_x=1 corner (P_CORNER2). There the texture input IS the
# noise and the model must predict E[x_0|cond] — the generation task itself, and
# joint's first step reads exactly that velocity. So P_CORNER2 is never masked.
#
# 0.4 (2026-08-11, warm start) -> 0.2 (2026-08-17, owner): back to MF's own
# value. Measured at 50k steps, tex|GT-mesh is the ONLY mode under MSE 1.0
# (0.971) while joint geo 1.167 / mesh-only 1.171 lag, so the freed 20% is worth
# more in the joint region than in the mode that is already ahead.
# ── (t_s, t_x) 采样：只有两个旋钮 ──────────────────────────────────────────
# The joint regime is ALWAYS uniform on the upper triangle (t_s <= t_x) — it is
# the DEFAULT DRAW in sample_timestep_pairs, not something you assemble out of
# probabilities. Every inference mode keeps t_s <= t_x, so t_s > t_x is a region
# that is never visited; the old parameterisation reached the triangle only if
# P_CORNER + P_CORNER2 + P_LAG happened to sum to 1, and the 2026-08-16 run
# missed it — 33% of every geometry-supervised sample landed at t_s > t_x.
# P_LAG is deleted: the triangle is now structural.
#
# These two only PIN THE TWO EDGES of that triangle; whatever is left over is
# the triangle's interior.
P_CORNER="${P_CORNER:-0.2}"   # t_s=0 edge: texture | given geometry, the flagship
                              # product mode. Geometry gets NO loss here, so this
                              # is pure cost to the geometry stream — 0.4 (the
                              # warm-start-era value) left it only 60% supervised.
P_CORNER2="${P_CORNER2:-0.2}" # t_x=1 edge: texture carries NOTHING. NOT raised above
                              # 0.2, even though that edge is "the inference
                              # regime": with rescale_t=3 the alpha=32 rollout has
                              # t_x > 0.9 for 11 of its 12 steps, but only step 1
                              # sits at t_x EXACTLY 1 — the other 10 live in the
                              # BAND (0.9, 1.0), which the triangle interior
                              # supplies, not this edge. Measured share of
                              # geometry-supervised samples landing in that band:
                              # 14% at 0.2/0.2/0.6 vs 11% at 0.2/0.3/0.5, i.e.
                              # raising this edge SHRINKS the region the
                              # trajectory actually occupies.
# Modality dropout, SYMMETRIC at 0.1 each (user 2026-08-18). They are mutually
# exclusive in flow_heads (`qdrop = rand < p & ~ddrop`), so full conditioning is
# 0.9 * 0.8 = 72% — the released pipeline has one dropout and 90%, but it has one
# conditioning source and we have two.
#   QWEN_DROP forces the flow onto DINO's 1029 patch tokens, the only spatially
#   precise signal we have and the one geometry needs.
#   DINO_DROP is the mirror. It is on symmetry grounds, NOT evidence: measured on
#   checkpoint-60000, removing DINO costs texture 11.4% and geometry only 3.5%,
#   i.e. the geometry stream is ALREADY ignoring it. 0.3 (what the 60K run
#   actually trained with) creates a third regime — "Qwen present, DINO absent" —
#   on 27% of samples, which CFG inference never uses. If the probe shows
#   cond_sens_geo flat, zero this first.
DINO_DROP="${DINO_DROP:-0.1}"
QWEN_DROP="${QWEN_DROP:-0.1}"
GC="${GC:-1.0}"               # STATIC fallback fraction (only used when ELASTIC=False)
ELASTIC="${ELASTIC:-True}"
ELASTIC_RATIO="${ELASTIC_RATIO:-0.75}"
UNFREEZE_GEO="${UNFREEZE_GEO:-False}"
GEO_LOSS_W="${GEO_LOSS_W:-1.0}"
DISTILL_W="${DISTILL_W:-1.0}"
BIDIR="${BIDIR:-True}"        # corner-masked bidirectional geo<->tex (user topology 2026-08-11)
FUSED="${FUSED:-True}"        # fused MMDiT attention: 1 native varlen call/lane; MFU 18.7%->30.0%
MIX="${MIX:-configs/mix_s3_splits.yaml}"
SHAPE_INIT="${SHAPE_INIT:-runs/s3_shape_t50b/checkpoint-8000}"
TEX_INIT="${TEX_INIT:-runs/s3_tex_t50b/checkpoint-8000}"
# ── FROM SCRATCH: three-stream sparse MMDiT (trellis2_blip3o/mmdit3d.py) ──
# No TRELLIS.2 DiT weights (the VAEs are unchanged); SHAPE_INIT/TEX_INIT are
# ignored, and so are the warm-start-only knobs (DISTILL_W, COND_MODE,
# COND_STREAM_BLOCKS, XATTN_ANNEAL_*, COUPLING, BIDIR). Sizing is these four:
FROM_SCRATCH="${FROM_SCRATCH:-False}"
DIM="${DIM:-768}"; HEADS="${HEADS:-6}"          # head_dim must leave a spare rope pair
DEPTH_DOUBLE="${DEPTH_DOUBLE:-8}"               # triple-stream blocks (own weights)
DEPTH_SINGLE="${DEPTH_SINGLE:-16}"              # shared-weight blocks (owner's 1:2)

if ! command -v torchrun >/dev/null 2>&1; then
  source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis
fi
cd "$(dirname "$0")/.."
export PATH="$(dirname "$(command -v python)"):$PATH"
if [ "$FROM_SCRATCH" != "True" ]; then
  [ -e "$SHAPE_INIT" ] || { echo "[geotex] SHAPE_INIT not found: $SHAPE_INIT" >&2; exit 1; }
  [ -e "$TEX_INIT" ]   || { echo "[geotex] TEX_INIT not found: $TEX_INIT" >&2; exit 1; }
fi
[ -e "$MIX" ]        || { echo "[geotex] MIX not found: $MIX" >&2; exit 1; }

# node-local caches (Lustre mmap crash — same rationale as the split launcher)
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/ind_geotex}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/dev/shm/tri_geotex}"
export TMPDIR="${TMPDIR:-/dev/shm/tmpc_geotex}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR" 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
# production kernel stack — the SAME kernels G0 certified bit-exactness through
export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn_3}"
export FUSED_MODULATE="${FUSED_MODULATE:-1}"
# the t50b ckpts' fixed sincos view table was built at scale 0.2
export VIEW_EMBED_SCALE="${VIEW_EMBED_SCALE:-0.2}"

EFF_BS="${EFF_BS:-256}"
NNODES="${NNODES:-1}"
WORLD=$(( PER_GPU_BS * NPROC * NNODES ))
GA=$(( EFF_BS / WORLD ))
[ $(( GA * WORLD )) -eq "$EFF_BS" ] || {
  echo "[geotex] PER_GPU_BS($PER_GPU_BS) × NPROC($NPROC) × NNODES($NNODES) must divide EFF_BS($EFF_BS)" >&2; exit 1; }

# multi-node: same contract and NCCL/EFA env as scripts/train_native_split.sh:64-75
# (NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT). Launch with scripts/launch_geotex_2n.sh.
DIST_FLAGS=""
if [ "$NNODES" -gt 1 ]; then
  DIST_FLAGS="--nnodes=$NNODES --node-rank=${NODE_RANK:?NODE_RANK required for NNODES>1} \
              --master-addr=${MASTER_ADDR:?} --master-port=${MASTER_PORT:-29601}"
  export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
  export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_ASYNC_ERROR_HANDLING=1
  _IF="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
  export NCCL_SOCKET_IFNAME="$_IF" GLOO_SOCKET_IFNAME="$_IF"
fi

DS_CFG="${DEEPSPEED:-configs/deepspeed_zero1_fp32acc.json}"
if [ "$DS_CFG" = "none" ]; then DS_FLAG="--ddp_find_unused_parameters False";
else DS_FLAG="--deepspeed $DS_CFG"; fi
REPORT="${REPORT_TO:-wandb}"
OUT="runs/geotex_s1${RUN_TAG:+_$RUN_TAG}"
echo "[geotex] nproc=$NPROC bs=$PER_GPU_BS ga=$GA (eff $EFF_BS) coupling=$COUPLING"
echo "         corner=$P_CORNER corner2=$P_CORNER2 mix=$MIX out=$OUT"
echo "         shape=$SHAPE_INIT tex=$TEX_INIT"

torchrun --nproc_per_node="$NPROC" ${DIST_FLAGS:-} train_native.py \
  --vlm_model Qwen/Qwen3.5-2B \
  --mixture_config "$MIX" \
  --build_vlm False \
  --fuse_dino True --dino_drop_prob "$DINO_DROP" --qwen_drop_prob "$QWEN_DROP" \
  --cond_max_length 10240 \
  --train_stages geotex \
  --geotex_shape_init "$SHAPE_INIT" --geotex_tex_init "$TEX_INIT" \
  --geotex_from_scratch "$FROM_SCRATCH" \
  --geotex_dim "$DIM" --geotex_heads "$HEADS" \
  --geotex_mlp_ratio "${MLP_RATIO:-5.3334}" --geotex_init "${INIT:-scaled}" \
  --geotex_depth_double "$DEPTH_DOUBLE" --geotex_depth_single "$DEPTH_SINGLE" \
  --geotex_coupling "$COUPLING" --geotex_cond_mode "$COND_MODE" \
  --geotex_cond_stream_blocks "$COND_STREAM_BLOCKS" \
  --geotex_xattn_anneal_start "$XATTN_ANNEAL_START" --geotex_xattn_anneal_end "$XATTN_ANNEAL_END" \
  --geotex_bidir "$BIDIR" --geotex_fused "$FUSED" --geotex_gc "$GC" \
  --geotex_unfreeze_geo "$UNFREEZE_GEO" --geotex_geo_loss_w "$GEO_LOSS_W" \
  --geotex_distill_w "$DISTILL_W" \
  --geotex_p_corner "$P_CORNER" --geotex_p_corner2 "$P_CORNER2" \
  --build_slat False --ss_only False --compile_ss_flow False \
  --elastic_slat "$ELASTIC" --elastic_target_ratio "$ELASTIC_RATIO" \
  --freeze_vlm True --flow_tune full \
  --cond_adapter mlp --cond_pos_stamp False --view_embed_mode sincos \
  --target_tokens_per_view 1024 --slat_resolution 512 --cond_fusion none \
  --output_dir "$OUT" \
  --max_steps "$MAX_STEPS" --bf16 True \
  --per_device_train_batch_size "$PER_GPU_BS" --gradient_accumulation_steps "$GA" \
  --learning_rate "$LR" --warmup_steps "${WARMUP:-100}" --weight_decay 0.01 \
  --lr_scheduler_type "${LR_SCHED:-constant_with_warmup}" \
  --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 \
  --ema_decay "${EMA:-0.9999}" \
  --logging_steps 5 --save_steps "${SAVE_STEPS:-1000}" --save_total_limit "${SAVE_KEEP:-4}" \
  --report_to "$REPORT" ${DS_FLAG} \
  --ignore_data_skip True --dataloader_num_workers "$WORKERS" ${EXTRA_ARGS:-}
