#!/bin/zsh
# 통합 워처: pt78(stage B, zip 마커) + oof74(OOF 모드, npy 마커) — 상태변화만 이벤트.
SSH_OPTS=(-i ~/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15)
HOST=<REMOTE_USER>@<CHEETAH_IP>
prev=""
for i in {1..60}; do
  cur=$(ssh $SSH_OPTS $HOST '
    now=$(date +%s)
    z="-"; [ -f ~/dacon/run_kf768pt78/submit_base2.zip ] && z=ZIP
    h=$(grep -h "홀드아웃 Macro-F1" ~/dacon/run_kf768pt78/train.log 2>/dev/null | tail -1 | sed "s/.*= //;s/ .*//")
    e1=$(grep -cE "Traceback|CUDA out of memory" ~/dacon/run_kf768pt78/train.log 2>/dev/null)
    a1=$(( now - $(stat -c %Y ~/dacon/run_kf768pt78/train.log 2>/dev/null || echo 0) ))
    s1=RUN; [ "$z" = ZIP ] && s1=DONE; [ "$z" != ZIP ] && [ "$a1" -gt 900 ] && s1=STALLED
    o="-"; [ -f ~/dacon/run_oof74/oof_logits_seed74.npy ] && o=NPY
    f=$(grep -h "\[OOF\] fold" ~/dacon/run_oof74/train.log 2>/dev/null | tail -1 | sed "s/.*fold //;s/ .*//")
    e2=$(grep -cE "Traceback|CUDA out of memory" ~/dacon/run_oof74/train.log 2>/dev/null)
    a2=$(( now - $(stat -c %Y ~/dacon/run_oof74/train.log 2>/dev/null || echo 0) ))
    s2=RUN; [ "$o" = NPY ] && s2=DONE; [ "$o" != NPY ] && [ "$a2" -gt 900 ] && s2=STALLED
    echo "pt78 st=$s1 holdout=${h:--} err=$e1 ; oof74 st=$s2 fold=${f:--} err=$e2"' 2>/dev/null)
  if [ -n "$cur" ] && [ "$cur" != "$prev" ]; then
    echo "[$(date '+%H:%M')] $cur"
    prev="$cur"
  fi
  done_n=$(echo "$cur" | grep -o "st=DONE" | wc -l | tr -d " ")
  bad=$(echo "$cur" | grep -cE "st=STALLED|err=[1-9]")
  if [ "${done_n:-0}" -eq 2 ]; then echo "ALL COMPLETE"; exit 0; fi
  if [ "${bad:-0}" -gt 0 ]; then echo "PROBLEM"; exit 1; fi
  sleep 300
done
echo TIMEOUT
exit 1
