#!/usr/bin/env bash
# Per-NODE stage: pack THIS node's group(s) of ready_v2 directly from the /fsx cond cache
# into WebDataset .tar shards on LOCAL NVMe. Run once on every training node at job start.
#   • 4 nodes → each node packs 1 group (~775GB, ~8-10min, all in parallel)
#   • 1 node  → packs all 4 groups (~3.1TB, ~30min)
# No intermediate shard copy on /fsx (which is ~94% full). Resumable (skips shards already
# on local NVMe). Group of shard i = i % NUM_GROUPS; node owns groups {g : g%NUM_NODES==rank}.
#
# Node rank/count come from SLURM (SLURM_NODEID / SLURM_NNODES) or NODE_RANK/NUM_NODES env.
set -euo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
NUM_GROUPS="${NUM_GROUPS:-4}"
OUT="${OUT:-/opt/dlami/nvme/weikaih_wds}"
MANIFEST="${MANIFEST:-/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v2/ready_v2_clean.jsonl}"
PROCS="${PROCS:-32}"
echo "[pack-node] node $NODE_RANK/$NUM_NODES groups-of-$NUM_GROUPS → $OUT ($(df -h "$(dirname "$OUT")" | tail -1 | awk '{print $4}') free)"
python3 scripts/pack_webdataset.py \
  --manifest "$MANIFEST" --out "$OUT" --shard-size 1000 --procs "$PROCS" \
  --num-groups "$NUM_GROUPS" --num-nodes "$NUM_NODES" --node-rank "$NODE_RANK"
echo "[pack-node] done: $(ls "$OUT"/*.tar 2>/dev/null | wc -l) shards, $(du -sh "$OUT" 2>/dev/null | cut -f1) on local NVMe"
