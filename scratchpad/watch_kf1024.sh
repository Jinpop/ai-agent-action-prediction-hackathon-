#!/bin/zsh
# Watch cheetah kf1024 refit completion (marker: run_X/submit_base2.zip).
# Polls every 4 min via fresh ssh (resilient to drops). Exits 0 when BOTH done.
SSH_OPTS=(-i ~/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15)
HOST=<REMOTE_USER>@<CHEETAH_IP>
for i in {1..45}; do
  out=$(ssh $SSH_OPTS $HOST 'ls ~/dacon/run_kf1024/submit_base2.zip 2>/dev/null; ls ~/dacon/run_kf1024_78/submit_base2.zip 2>/dev/null' 2>/dev/null)
  n=$(echo "$out" | grep -c submit_base2.zip)
  echo "[$(date '+%H:%M')] poll $i: $n/2 complete"
  if [ "$n" -eq 2 ]; then
    echo "BOTH kf1024 refits COMPLETE"
    ssh $SSH_OPTS $HOST 'grep -h "홀드아웃 Macro-F1" ~/dacon/run_kf1024/train.log ~/dacon/run_kf1024_78/train.log | tail -4'
    exit 0
  fi
  sleep 240
done
echo "TIMEOUT after 45 polls (~3h)"
exit 1
