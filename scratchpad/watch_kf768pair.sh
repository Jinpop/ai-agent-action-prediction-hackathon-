#!/bin/zsh
# Watch run_kf768ci74 + run_kf768_79 on cheetah — v2 (00:31 검토 반영).
# per-run 판정: zip 존재=완료 / train.log mtime 15분 무갱신 & zip 없음=크래시.
# pgrep 사용 안 함(런처 셸·원격 셸 오탐 — 00:31 검토 지적).
SSH_OPTS=(-i ~/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15)
HOST=<REMOTE_USER>@<CHEETAH_IP>
prev=""
for i in {1..48}; do
  cur=$(ssh $SSH_OPTS $HOST '
    now=$(date +%s)
    for r in run_kf768ci74 run_kf768_79; do
      z="-"; [ -f ~/dacon/$r/submit_base2.zip ] && z="ZIP"
      h=$(grep -h "홀드아웃 Macro-F1" ~/dacon/$r/train.log 2>/dev/null | tail -1 | sed "s/.*= //;s/ .*//")
      e=$(grep -cE "Traceback|CUDA out of memory|RuntimeError" ~/dacon/$r/train.log 2>/dev/null)
      age=$(( now - $(stat -c %Y ~/dacon/$r/train.log 2>/dev/null || echo 0) ))
      st="RUN"; [ "$z" = "ZIP" ] && st="DONE"
      [ "$z" != "ZIP" ] && [ "$age" -gt 900 ] && st="STALLED"
      echo "$r st=$st holdout=${h:--} err=$e logage=${age}s"
    done' 2>/dev/null)
  key=$(echo "$cur" | sed 's/ logage=[0-9-]*s//')   # 지터 필드 제외한 비교 키
  if [ -n "$cur" ] && [ "$key" != "$prev" ]; then
    echo "[$(date '+%H:%M')] $cur" | tr '\n' ' ; '; echo ""
    prev="$key"
  fi
  done_n=$(echo "$cur" | grep -c "st=DONE")
  bad_n=$(echo "$cur" | grep -cE "st=STALLED")
  err_n=$(echo "$cur" | grep -c "err=[1-9]")
  if [ "$done_n" -eq 2 ]; then echo "BOTH COMPLETE"; exit 0; fi
  if [ "$bad_n" -gt 0 ] || [ "$err_n" -gt 0 ]; then echo "PROBLEM: stalled=$bad_n err_runs=$err_n"; exit 1; fi
  sleep 300
done
echo "TIMEOUT ~4h"
exit 1
