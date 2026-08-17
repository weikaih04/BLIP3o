#!/bin/bash
# 只在"需要动作"时输出:进程死了、或步数卡住。正常推进时保持沉默。
LOG=/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/geotex_v5_655m_r0.log
step() { tail -c 600 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/60000' | tail -1 | cut -d/ -f1; }
prev=$(step); same=0
while true; do
  sleep 120
  alive=$(tmux ls 2>/dev/null | grep -c geotex_v5_655m)
  cur=$(step)
  if [ "${alive:-0}" -lt 4 ]; then
    echo "ALERT 训练挂了:tmux 只剩 ${alive}/4 个会话,最后步数 ${cur:-?}/60000"
    grep -v '— resampling' "$LOG" | grep -iE 'Traceback|Error|FAILED|Killed|exitcode' | grep -viE 'NCCL_ASYNC|deprecated' | tail -3
    exit 1
  fi
  if [ "$cur" = "$prev" ]; then
    same=$((same+1))
    [ "$same" -ge 3 ] && { echo "ALERT 训练卡住:步数 6 分钟停在 ${cur}/60000,4 个会话都还在"; exit 1; }
  else
    same=0; prev=$cur
  fi
done
