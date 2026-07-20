#!/bin/bash
# campaign official-meta-wave1 META-N/T 완료감시 (재접속 폴링)
# MARK<dN><rN>-<dT><rT>: 01=running 10=done 00=crash. r는 실제 chain bash만(grep 필터로 pgrep 래퍼 자기매칭 배제)
SSH="ssh -i /Users/<USER>/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes <REMOTE_USER>@<CHEETAH_IP>"
REMOTE='dN=$(grep -c ALL_DONE ~/dacon/run_metaN_s79/chain.status 2>/dev/null); dN=${dN:-0};
dT=$(grep -c ALL_DONE ~/dacon/run_metaT_s79/chain.status 2>/dev/null); dT=${dT:-0};
rN=$(pgrep -af "run_metaN_chain.sh" 2>/dev/null | grep -c "bash run_metaN_chain.sh"); rN=${rN:-0};
rT=$(pgrep -af "run_metaT_chain.sh" 2>/dev/null | grep -c "bash run_metaT_chain.sh"); rT=${rT:-0};
echo "MARK${dN}${rN}-${dT}${rT}"'
END=$(( $(date +%s) + 14400 ))   # 백스톱 ~4h (REFIT_ONLY 6ep ~2h ×2)
echo "METAWAVE_WATCH_START $(date '+%F %H:%M:%S') (poll 120s, backstop +4h)"
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
