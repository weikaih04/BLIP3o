#!/bin/bash
# 2-node geotex training — same shape as scripts/launch_im_ss_2n.sh (env-prefixed
# per-rank launch; the NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT contract and the
# NCCL/EFA env live in scripts/train_native_geotex.sh, copied verbatim from
# train_native_split.sh:64-75).
#
# TWO differences from launch_im_ss_2n.sh, both forced by the assigned-hold model
# (skill xgen-mm-chi-aws §0) — holds are ONE NODE EACH, and long runs cannot use
# sbatch without opening a 5th allocation:
#
#   1. Each rank is srun'd into its OWN jobid instead of `-w` inside one
#      allocation. torchrun's rendezvous is plain TCP, so the allocation boundary
#      is invisible to it.
#   2. Each rank's srun CLIENT lives in a tmux session ON THE LOGIN NODE. A bare
#      `srun --overlap` launched from an agent shell is reaped when that shell
#      goes away — it killed S2b at step 726 on 2026-08-13, no traceback, no
#      checkpoint (memory: long-jobs-need-sbatch; scripts/geotex_s1.sbatch is why
#      S1 survived 7h). tmux ON THE COMPUTE NODE does NOT work here: `tmux
#      new-session -d` inside an --overlap step never brings up a server
#      (measured 2026-08-13, socket /tmp/tmux-<uid>/default never appears), so
#      the skill's §0.1.1 recipe does not apply to this cluster's overlap steps.
#      The login-node tmux server does persist (sessions there survive for days),
#      and an idle srun client costs a few MB — well inside the 9 GB login cgroup.
#      Watch with:  tmux attach -t geotex_<TAG>_r0
#
# Before running: stop the Molmo2 benchmark STEP on any hold you are taking over
# (`scancel <job>.<step>` — never `scancel <job>`), wait for its GPUs to drain,
# and launch IMMEDIATELY: the supervisor starts a new benchmark round after 20
# min of full idle and will OOM your ranks (memory: placeholder-hold-20min-window).
#
# Usage: JOB0=1127 JOB1=1148 RUN_TAG=s2b UNFREEZE_GEO=True \
#          SHAPE_INIT=... TEX_INIT=... bash scripts/launch_geotex_2n.sh
set -u
cd /fsx/home/weikai.huang/3dgen/model/BLIP3o
ROOT="$PWD"
# N nodes: HOLDS="1247 1248 1253 1151" (rank order = listed order, rank 0 is
# the rendezvous master). JOB0/JOB1 still work for the 2-node case.
HOLDS="${HOLDS:-${JOB0:?set HOLDS='j0 j1 ...' or JOB0/JOB1} ${JOB1:?}}"
PORT="${MASTER_PORT:-29613}"
TAG="${RUN_TAG:-2n}"
SESS="geotex_${TAG}"

read -r -a JOBS <<< "$HOLDS"
NNODE=${#JOBS[@]}
N=()
for j in "${JOBS[@]}"; do
  n="$(squeue -j "$j" -h -o %N)"
  [ -n "$n" ] || { echo "[geotex-Nn] cannot resolve node for job $j" >&2; exit 1; }
  N+=("$n")
done
mkdir -p runs/cache_logs

# WORKLOAD RESERVATION — the supervisor's own opt-out
# (adacodec_vlm/runtime/molmo2_stage2_eval_supervisor.sh:37,333-341). Without
# it the Molmo2 supervisor reclaims the node: on 2026-08-15 it SIGKILLed all 8
# ranks of a run 10 minutes in, no traceback, exitcode -9. scancel'ing its step
# is NOT enough — it respawns within 24 s. Reserve, then launch.
RESV=/fsx/home/weikai.huang/adacodec_vlm/logs/molmo2_stage2_eval
for r in $(seq 0 $((NNODE-1))); do
  touch "$RESV/workload_${JOBS[r]}__${N[r]}.reserved" 2>/dev/null \
    && echo "[geotex-Nn] reserved ${N[r]} (job ${JOBS[r]})"
done
echo "[geotex-Nn] NOTE: rm $RESV/workload_*.reserved when the run is done, or"
echo "            those nodes never run benchmark again."
sleep 45   # give each supervisor a poll cycle to stop its runner

# HARD GATE — refuse to launch into occupied GPUs. On 2026-08-13 a drain loop
# that only `break`s on success fell through after 30 tries and launched anyway;
# the Molmo2 benchmark had retaken one node (28.89 GiB/GPU) and every rank OOM'd
# six minutes in. Free memory is the precondition, not a hope.
for r in $(seq 0 $((NNODE-1))); do
  USED=$(srun --jobid="${JOBS[r]}" --overlap --nodes=1 -w "${N[r]}" --ntasks=1 --cpus-per-task=1 \
           nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk '{s+=$1} END{print s+0}')
  if [ "${USED:-999999}" -ge 4000 ]; then
    echo "[geotex-2n] ABORT: ${N[r]} (job ${JOBS[r]}) has ${USED} MiB of GPU memory in use." >&2
    echo "            Stop the benchmark STEP (scancel <job>.<step>, never the job), wait for" >&2
    echo "            it to drain, and rerun — the supervisor needs 20 min idle to come back." >&2
    exit 1
  fi
done
FWD=""
# ⚠️ tmux starts a FRESH shell that inherits nothing, so anything not named here
# silently falls back to the launcher's default. 2026-08-15: COND_STREAM_BLOCKS
# and the anneal window were missing, so a run meant to be 30-block + annealed
# quietly trained the 10-block un-annealed config instead. Add new knobs HERE.
for v in MAX_STEPS PER_GPU_BS EFF_BS WORKERS REPORT_TO SAVE_STEPS SAVE_KEEP LR \
         UNFREEZE_GEO GEO_LOSS_W DISTILL_W SHAPE_INIT TEX_INIT MIX DINO_DROP QWEN_DROP \
         P_CORNER P_CORNER2 COUPLING COND_MODE BIDIR FUSED \
         COND_STREAM_BLOCKS XATTN_ANNEAL_START XATTN_ANNEAL_END \
         FROM_SCRATCH DIM HEADS DEPTH_DOUBLE DEPTH_SINGLE MLP_RATIO INIT \
         EMA LR_SCHED WARMUP \
         WANDB_PROJECT WANDB_NAME WANDB_ENTITY \
         GC ELASTIC ELASTIC_RATIO DEEPSPEED EXTRA_ARGS; do
  [ -n "${!v:-}" ] && FWD="$FWD $v=$(printf %q "${!v}")"
done
echo "[geotex-Nn] ${NNODE} nodes: ${N[*]} (jobs ${JOBS[*]}) master=${N[0]}:$PORT tmux=$SESS"
echo "[geotex-Nn] forwarded:$FWD"   # verify what actually crosses into tmux

# Everything the run needs, forwarded explicitly — tmux starts a fresh shell that
# does NOT inherit this one's environment.

for r in $(seq 0 $((NNODE-1))); do
  # per-rank runner file — avoids nested quoting inside the tmux command string
  RUN="runs/cache_logs/.run_${TAG}_r${r}.sh"
  {
    echo "#!/bin/bash"
    echo "cd $ROOT"
    # MASTER_ADDR is a HOSTNAME on purpose — ablate_node.sh:5-9 records that
    # getent returns multiline IPv6 on these compute nodes; torchrun resolves it.
    echo "exec env NNODES=$NNODE NODE_RANK=$r MASTER_ADDR=${N[0]} MASTER_PORT=$PORT RUN_TAG=$TAG$FWD \\"
    echo "  bash scripts/train_native_geotex.sh 8 > runs/cache_logs/geotex_${TAG}_r${r}.log 2>&1"
  } > "$RUN"
  chmod +x "$RUN"
  tmux kill-session -t "${SESS}_r${r}" 2>/dev/null
  tmux new-session -d -s "${SESS}_r${r}" \
    "srun --jobid=${JOBS[r]} --overlap --nodes=1 -w ${N[r]} --ntasks=1 --cpus-per-task=90 bash $ROOT/$RUN"
  echo "[geotex-2n] rank$r launched via login-node tmux '${SESS}_r${r}' → ${N[r]}"
done
echo "[geotex-2n] logs runs/cache_logs/geotex_${TAG}_r{0,1}.log"
