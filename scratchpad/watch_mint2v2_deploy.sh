#!/bin/bash
# campaign mint2v2-dualseed-0714 DEPLOY 완료감시 (재접속 폴링)
# 상태 페어 MARK<d79><r79>-<d81><r81>: 01=running 10=done 00=crash
# r는 실제 chain bash만 카운트(pgrep 래퍼 자기매칭 배제 위해 grep "bash run..." 로 필터)
SSH="ssh -i /Users/<USER>/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes <REMOTE_USER>@<CHEETAH_IP>"
REMOTE='d79=$(grep -c ALL_DONE ~/dacon/run_mint2v2dep_s79/chain.status 2>/dev/null); d79=${d79:-0};
d81=$(grep -c ALL_DONE ~/dacon/run_mint2v2dep_s81/chain.status 2>/dev/null); d81=${d81:-0};
r79=$(pgrep -af "run_mint2v2dep_chain.sh 79" 2>/dev/null | grep -c "bash run_mint2v2dep_chain.sh 79"); r79=${r79:-0};
r81=$(pgrep -af "run_mint2v2dep_chain.sh 81" 2>/dev/null | grep -c "bash run_mint2v2dep_chain.sh 81"); r81=${r81:-0};
echo "MARK${d79}${r79}-${d81}${r81}"'
END=$(( $(date +%s) + 14400 ))   # 백스톱 ~4h (deploy ETA ~2h ×2)
echo "DEPLOY_WATCH_START $(date '+%F %H:%M:%S') (poll 120s, backstop +4h)"
while true; do
  out=$($SSH "$REMOTE" 2>/dev/null)
  mark=$(printf '%s' "$out" | grep -oE 'MARK[0-9]+-[0-9]+' | tail -1)
  ts=$(date '+%F %H:%M:%S')
  case "$mark" in
    MARK1?-1?)     echo "[$ts] BOTH_DONE $mark"; break ;;
    MARK00-*|*-00) echo "[$ts] CRASH_SUSPECT $mark"; break ;;
    '')            echo "[$ts] poll-empty(net blip, retry)" ;;
    *)             echo "[$ts] running $mark" ;;
  esac
  if [ "$(date +%s)" -gt "$END" ]; then echo "[$ts] BACKSTOP still-running $mark -- 재부착 필요"; break; fi
  sleep 120
done
echo "WATCHER_EXIT mark=$mark $(date '+%F %H:%M:%S')"
