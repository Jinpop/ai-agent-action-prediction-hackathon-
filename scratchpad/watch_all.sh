#!/bin/zsh
# 통합 워처(갱신): 치타 ec79o(v56 zip) + H100 h83/h84(refit zip). h79v2는 의도적 중단이라 제외.
# 상태변화·완료·문제만 이벤트. 실패 커버: Traceback/OOM/mtime STALL.
CH=(-i ~/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15)
CHH=<REMOTE_USER>@<CHEETAH_IP>
EL=(-i ~/.ssh/elice.pem -p <ELICE_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes)
ELH=elicer@<ELICE_TUNNEL_HOST>
prev=""
for i in {1..50}; do
  ch=$(ssh $CH $CHH '
    now=$(date +%s)
    z=-; [ -f ~/dacon/run_kf768_ec79o/submit_base2.zip ] && z=ZIP
    e=$(grep -cE "Traceback|CUDA out of memory" ~/dacon/run_kf768_ec79o/train.log 2>/dev/null)
    a=$(( now - $(stat -c %Y ~/dacon/run_kf768_ec79o/train.log 2>/dev/null || echo 0) ))
    s=RUN; [ "$z" = ZIP ] && s=DONE; [ "$z" != ZIP ] && [ "$a" -gt 900 ] && s=STALL
    st=$(tail -c 1500 ~/dacon/run_kf768_ec79o/train.log 2>/dev/null|tr "\r" "\n"|grep -oE "[0-9]+/[0-9]+"|tail -1)
    echo "ec79o=$s:${st:--}(e$e)"' 2>/dev/null)
  el=$(ssh $EL $ELH '
    now=$(date +%s); out=""
    for R in run_h83 run_h84; do
      d=~/dacon/$R
      st=RUN; [ -f $d/submit_base2.zip ] && st=DONE
      if [ "$st" != DONE ] && grep -qE "Traceback|CUDA out of memory" $d/train.log 2>/dev/null; then st=ERR; fi
      a=$(( now - $(stat -c %Y $d/train.log 2>/dev/null || echo 0) ))
      [ "$st" = RUN ] && [ "$a" -gt 900 ] && st=STALL
      step=$(tail -c 2500 $d/train.log 2>/dev/null|tr "\r" "\n"|grep -oE "[0-9]+/9150"|tail -1)
      out="$out ${R#run_}=$st:${step:--}"
    done
    echo "procs=$(pgrep -f colab_train_base2|wc -l|tr -d " ")$out"' 2>/dev/null)
  cur="CH[$ch] EL[$el]"
  if [ -n "$el" -o -n "$ch" ] && [ "$cur" != "$prev" ]; then echo "[$(date +%H:%M)] $cur"; prev="$cur"; fi
  bad=$(echo "$cur" | grep -cE "=ERR|=STALL")
  chdone=$(echo "$ch" | grep -c "=DONE")
  eldone=$(echo "$el" | grep -oE "=DONE" | wc -l | tr -d " ")
  if [ "${chdone:-0}" -ge 1 ] && [ "${eldone:-0}" -ge 2 ]; then echo "ALL COMPLETE"; exit 0; fi
  if [ "${bad:-0}" -gt 0 ]; then echo "PROBLEM: $cur"; exit 1; fi
  sleep 240
done
echo TIMEOUT
