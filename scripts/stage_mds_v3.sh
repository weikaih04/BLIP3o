#!/bin/bash
# Per-node MDS staging for ready_v3 (node-sharded into NUM_SHARDS=4).
# Each node builds ONLY the shards it owns (shard_id % NNODES == NODE_RANK) from the /fsx
# source (cond cache + latents) to LOCAL NVMe. /fsx is too full (92%) to hold a 6TB MDS, so
# we never persist the MDS — each node restages its 1/N at job start. Idempotent: a shard with
# index.json present is skipped (survives low-pri requeue onto the SAME node; rebuilds on a new one).
#
#   4 nodes → 1 shard/node (~1.5TB, ~30min)   2 nodes → 2 shards/node (~1h)   1 node → all 4 (~1.9h)
#
# Run ONCE before training, on EVERY node:  srun --ntasks-per-node=1 bash scripts/stage_mds_v3.sh
set -uo pipefail
PY=/fsx/sfr/weikaih/miniconda3/envs/blip3o_trellis/bin/python
MANI=/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v3/ready_v3_clean.jsonl
DEST=${MDS_V3_ROOT:-/opt/dlami/nvme/weikaih_mds_v3_shards}
NUM_SHARDS=${MDS_NUM_SHARDS:-4}
NNODES=${SLURM_NNODES:-1}
RANK=${SLURM_NODEID:-0}
PROCS=${MDS_PROCS:-32}
mkdir -p "$DEST"
echo "[stage-mds] node $RANK/$NNODES  build shards k%$NNODES==$RANK  (of $NUM_SHARDS)  → $DEST"
for K in $(seq 0 $((NUM_SHARDS-1))); do
  [ $((K % NNODES)) -eq "$RANK" ] || continue
  OUT="$DEST/shard$K"
  if [ -f "$OUT/index.json" ]; then echo "[stage-mds] shard$K built already → skip"; continue; fi
  echo "[stage-mds] building shard$K → $OUT"
  "$PY" /fsx/sfr/weikaih/3dgen/model/BLIP3o/scripts/build_mds.py \
    --out "$OUT" --manifest "$MANI" --shard "$K" --num-shards "$NUM_SHARDS" --procs "$PROCS"
done
echo "[stage-mds] node $RANK done. owned shards on NVMe:"; ls -d "$DEST"/shard*/ 2>/dev/null
