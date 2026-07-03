#!/bin/bash
# Complete the v3 cond-cache MERGE phase (the ~1k that failed when /fsx filled up).
# Re-scans all need_cache v-files (resumable/idempotent: skips already-merged), merges the missing.
set -uo pipefail
source /fsx/sfr/weikaih/miniconda3/bin/activate blip3o_trellis
cd /fsx/sfr/weikaih/3dgen/model/BLIP3o
OUT=/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/qwen35-2b_tok1024_crop_mv1
MAN=/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v3/ready_v3_needcache.jsonl
LOGD=runs/cache_logs/v3merge2
mkdir -p "$LOGD"; rm -f "$LOGD/STATUS"
echo "[merge-v3] launching 32 CPU shards" > "$LOGD/main.log"
for i in $(seq 0 31); do
  python scripts/merge_vd_cache.py --root "$OUT" --manifest "$MAN" --max_views 1 \
    --shard "$i" --num_shards 32 > "$LOGD/m_$i.log" 2>&1 &
done
wait
echo "MERGE_DONE" > "$LOGD/STATUS"
echo "[merge-v3] ALL 32 shards done" >> "$LOGD/main.log"
