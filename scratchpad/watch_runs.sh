#!/bin/zsh
# Usage: watch_runs.sh <run1> <run2> ...  — cheetah run 감시 (v3 로직 일반화).
# per-run: zip=DONE / log 15분 무갱신=STALLED / Traceback 카운트. 상태변화만 이벤트.
RUNS=("$@")
SSH_OPTS=(-i ~/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15)
HOST=<REMOTE_USER>@<CHEETAH_IP>
prev=""
for i in {1..60}; do
  cur=$(ssh $SSH_OPTS $HOST "
    now=\$(date +%s)
    for r in ${RUNS[@]}; do
      z='-'; [ -f ~/dacon/\$r/submit_base2.zip ] && z=ZIP
      h=\$(grep -h '홀드아웃 Macro-F1' ~/dacon/\$r/train.log 2>/dev/null | tail -1 | sed 's/.*= //;s/ .*//')
      e=\$(grep -cE 'Traceback|CUDA out of memory|RuntimeError' ~/dacon/\$r/train.log 2>/dev/null)
      age=\$(( now - \$(stat -c %Y ~/dacon/\$r/train.log 2>/dev/null || echo 0) ))
      st=RUN; [ \"\$z\" = ZIP ] && st=DONE
      [ \"\$z\" != ZIP ] && [ \"\$age\" -gt 900 ] && st=STALLED
      echo \"\$r st=\$st holdout=\${h:--} err=\$e logage=\${age}s\"
    done" 2>/dev/null)
  key=$(echo "$cur" | sed 's/ logage=[0-9-]*s//')
  if [ -n "$cur" ] && [ "$key" != "$prev" ]; then
    echo "[$(date '+%H:%M')] $cur" | tr '\n' ' ; '; echo ""
    prev="$key"
  fi
  done_n=$(echo "$cur" | grep -c "st=DONE")
  bad_n=$(echo "$cur" | grep -c "st=STALLED")
  err_n=$(echo "$cur" | grep -c "err=[1-9]")
  if [ "$done_n" -eq ${#RUNS[@]} ]; then echo "ALL COMPLETE"; exit 0; fi
  if [ "$bad_n" -gt 0 ] || [ "$err_n" -gt 0 ]; then echo "PROBLEM: stalled=$bad_n err_runs=$err_n"; exit 1; fi
  sleep 300
done
echo "TIMEOUT ~5h"
exit 1
