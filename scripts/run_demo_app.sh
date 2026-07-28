#!/usr/bin/env bash
# Launch the tri-modal 3D demo on a held GPU node and expose it through a Cloudflare tunnel.
#
# This is the command registered with the viz hub (`hub.py add-app`), so `bash start_apps.sh`
# recovers the demo on ANY host: the script finds a RUNNING held job itself, steps into it
# with one GPU, and starts both the app and the tunnel INSIDE that job step (so the tunnel
# always points at localhost of the node the model is actually on).
#
#   bash scripts/run_demo_app.sh              # pick the first running held job
#   DEMO_JOBIDS="194" DEMO_PORT=7861 bash scripts/run_demo_app.sh
#
# Tunnel: if a named-tunnel config exists (stable hostname) it is used; otherwise the script
# falls back to a no-login "quick" tunnel, whose *.trycloudflare.com URL changes on every
# restart. Upgrading to a STABLE hostname needs one interactive browser login, which an agent
# cannot do. Run these once on any host (creds land in ~/.cloudflared, which is on shared
# /fsx, so every node picks them up and they are gitignored):
#
#   cloudflared tunnel login                       # opens a browser; pick ai-research-wk.com
#   cloudflared tunnel create blip3o-demo
#   cloudflared tunnel route dns blip3o-demo blip3o.ai-research-wk.com
#   cat > ~/.cloudflared/blip3o.yml <<'YML'
#   tunnel: blip3o-demo
#   credentials-file: /fsx/home/weikai.huang/.cloudflared/<TUNNEL-UUID>.json
#   ingress:
#     - hostname: blip3o.ai-research-wk.com
#       service: http://localhost:7860
#     - service: http_status:404
#   YML
#
# After that this script uses it automatically and the URL stops changing.
#
# NEVER scancel the held jobs — they are node holds, not this app. Stop the demo with
#   bash scripts/run_demo_app.sh stop
set -uo pipefail

ROOT=/fsx/home/weikai.huang/3dgen/model/BLIP3o
PORT=${DEMO_PORT:-7860}
JOBIDS=${DEMO_JOBIDS:-"193 194 202"}
CF=${CLOUDFLARED:-/fsx/home/weikai.huang/.local/bin/cloudflared}
TUNNEL_CFG=${DEMO_TUNNEL_CONFIG:-$HOME/.cloudflared/blip3o.yml}
CONDA=/fsx/home/weikai.huang/miniconda3/bin/activate
LOGDIR=$ROOT/runs
APP_LOG=$LOGDIR/demo_app.log
CF_LOG=$LOGDIR/demo_tunnel.log
URL_FILE=$LOGDIR/demo_url.txt
mkdir -p "$LOGDIR"

if [ "${1:-}" = "stop" ]; then
  pkill -f "scripts/demo_app.py" && echo "stopped app"
  pkill -f "cloudflared tunnel .*(blip3o|$PORT)" && echo "stopped tunnel"
  exit 0
fi

# ── pick a RUNNING held job (they are node holds; we only --overlap into them) ──
JOB=""
for j in $JOBIDS; do
  if squeue -h -j "$j" -o "%T" 2>/dev/null | grep -q RUNNING; then JOB=$j; break; fi
done
if [ -z "$JOB" ]; then
  echo "no RUNNING job among: $JOBIDS  (squeue -u \$USER)" >&2
  exit 1
fi
echo "[demo] using job $JOB, port $PORT"

: > "$APP_LOG"; : > "$CF_LOG"; : > "$URL_FILE"

srun --jobid="$JOB" --overlap --gres=gpu:1 --job-name=blip3o_demo \
  bash -c "
set -uo pipefail
cd $ROOT
source $CONDA blip3o_trellis
export HF_HOME=/fsx/home/weikai.huang/.cache/huggingface
export ATTN_BACKEND=flash_attn FUSED_MODULATE=1 TOKENIZERS_PARALLELISM=false
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRADIO_ANALYTICS_ENABLED=False

# tunnel first so the URL exists while the ~2 min cold start runs
if [ -f '$TUNNEL_CFG' ]; then
  echo '[demo] named tunnel from $TUNNEL_CFG'
  nohup setsid $CF tunnel --config '$TUNNEL_CFG' run >> '$CF_LOG' 2>&1 < /dev/null &
else
  echo '[demo] no named-tunnel config at $TUNNEL_CFG -> quick tunnel (URL changes on restart)'
  nohup setsid $CF tunnel --url http://localhost:$PORT --no-autoupdate \
      >> '$CF_LOG' 2>&1 < /dev/null &
fi
( for i in \$(seq 1 60); do
    u=\$(grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' '$CF_LOG' | head -1)
    if [ -n \"\$u\" ]; then echo \"\$u\" > '$URL_FILE'; break; fi
    sleep 2
  done ) &

echo \"[demo] host \$(hostname) gpu \${CUDA_VISIBLE_DEVICES:-?}\"
exec python scripts/demo_app.py --port $PORT
" >> "$APP_LOG" 2>&1
