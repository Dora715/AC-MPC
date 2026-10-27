#!/bin/bash
# A 方案：从 acmpc_splits 625k（80% 完赛态）续训 1M 步，测成功走廊能否稳住。
cd /root/acmpc
PY=/root/miniforge3/envs/acmpc/bin/python
name="acmpc_splits_run1_resumeA"
latest="/root/acmpc/runs/ckpt/$name/LATEST"
resume="/root/acmpc/runs/ckpt/acmpc_splits_run1/acmpc_splits_run1_625000"
rm -f "logs/$name.log"
seg=1
while true; do
  echo "[$(date +%F_%T)] SEG$seg START $name resume='$resume'"
  $PY train_race.py --track splits --policy mpc --total-timesteps 1625000 --tag run1_resumeA \
    --resume "$resume" >> "logs/$name.log" 2>&1
  ec=$?
  if [ "$ec" -eq 42 ] && [ -f "$latest" ]; then
    resume=$(cat "$latest")
    echo "[$(date +%F_%T)] SEG$seg END $name -> restart with $resume"
    seg=$((seg+1))
    continue
  fi
  echo "[$(date +%F_%T)] END $name (exit $ec)"
  break
done
