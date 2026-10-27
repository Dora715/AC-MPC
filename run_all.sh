#!/bin/bash
# AC-MPC vs AC-MLP 全量训练：3 赛道 × 2 策略，顺序执行（分段+重启版）
# train_race.py 每训练一段（默认 250k 步）后以退出码 42 结束，
# 本脚本循环重启新进程续训，绕过 mpc.pytorch LQRStep 的进程内 GPU 泄漏。
cd /root/acmpc
PY=/root/miniforge3/envs/acmpc/bin/python
mkdir -p logs

for pol in mpc mlp; do
  for track in horizontal vertical splits; do
    name="ac${pol}_${track}_run1"
    latest="/root/acmpc/runs/ckpt/$name/LATEST"
    resume=""
    rm -f "logs/$name.log"
    seg=1
    while true; do
      echo "[$(date +%F_%T)] SEG$seg START $name resume='$resume'"
      if [ -n "$resume" ]; then
        $PY train_race.py --track "$track" --policy "$pol" --total-timesteps 2000000 --tag run1 --resume "$resume" \
          >> "logs/$name.log" 2>&1
      else
        $PY train_race.py --track "$track" --policy "$pol" --total-timesteps 2000000 --tag run1 \
          >> "logs/$name.log" 2>&1
      fi
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
  done
done
echo "[$(date +%F_%T)] ALL DONE"
