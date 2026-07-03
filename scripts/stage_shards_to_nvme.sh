#!/usr/bin/env bash
# Stage packed WebDataset shards from /fsx (persistent) → local NVMe (fast, ephemeral)
# at train start. Idempotent (skips files already present with matching size). Parallel.
#   usage: bash scripts/stage_shards_to_nvme.sh [SRC] [DST] [PARALLEL]
set -euo pipefail
SRC="${1:-/fsx/sfr/weikaih/3dgen/data/webdataset_shards}"
DST="${2:-/opt/dlami/nvme/weikaih_wds}"
PAR="${3:-8}"
mkdir -p "$DST"
echo "[stage] $SRC → $DST  (parallel=$PAR)"
t0=$(date +%s)
# copy .idx.json + _wds_meta.json first (small, needed by the reader's index build)
cp -n "$SRC"/*.idx.json "$SRC"/_wds_meta.json "$DST"/ 2>/dev/null || true
# parallel copy the big .tar shards, skipping ones already fully present
ls "$SRC"/*.tar | xargs -P "$PAR" -I{} bash -c '
  f="{}"; b=$(basename "$f"); d="'"$DST"'/$b"
  if [ -f "$d" ] && [ "$(stat -c%s "$f")" = "$(stat -c%s "$d")" ]; then exit 0; fi
  cp "$f" "$d.tmp" && mv "$d.tmp" "$d"
'
n_src=$(ls "$SRC"/*.tar 2>/dev/null | wc -l); n_dst=$(ls "$DST"/*.tar 2>/dev/null | wc -l)
echo "[stage] done in $(( $(date +%s) - t0 ))s — $n_dst/$n_src shards on NVMe ($(du -sh "$DST" | cut -f1))"
