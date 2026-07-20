#!/bin/bash
# campaign mint2v2-dualseed-0714 gate 완료감시 (재접속 폴링 워처)
# zsh 단어분리 무관: 원격이 MARK<d79><r79>-<d81><r81> 프리픽스 echo, 로컬은 grep -oE + case 글로브로 매칭
# 상태 페어: 01=running(done0,chain1) 10=done(done1,chain0) 00=crash(done0,chain0)
SSH="ssh -i /Users/<USER>/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes <REMOTE_USER>@<CHEETAH_IP>"
REMOTE='d79=$(grep -c ALL_DONE ~/dacon/run_mint2v2b_s79/chain.status 2>/dev/null); d79=${d79:-0};
d81=$(grep -c ALL_DONE ~/dacon/run_mint2v2_s81_gate/chain.status 2>/dev/null); d81=${d81:-0};
r79=$(pgrep -fc run_mint2v2b_chain.sh 2>/dev/null); r79=${r79:-0};
r81=$(pgrep -fc run_mint2v2_s81gate_chain.sh 2>/dev/null); r81=${r81:-0};
echo "MARK${d79}${r79}-${d81}${r81}"'
END=$(( $(date +%s) + 12600 ))   # 백스톱 ~3h30m (gate ETA ~1h46m ×~2)
echo "WATCH_START $(date '+%F %H:%M:%S') (poll 110s, backstop +3h30m)"
while true; do
  out=$($SSH "$REMOTE" 2>/dev/null)
  mark=$(printf '%s' "$out" | grep -oE 'MARK[0-9]+-[0-9]+' | tail -1)
  ts=$(date '+%F %H:%M:%S')
  case "$mark" in
    MARK1?-1?) echo "[$ts] BOTH_DONE $mark"; break ;;
    MARK00-*|*-00)  echo "[$ts] CRASH_SUSPECT $mark"; break ;;
    '')        echo "[$ts] poll-empty(net blip, retry)" ;;
    *)         echo "[$ts] running $mark" ;;
  esac
  if [ "$(date +%s)" -gt "$END" ]; then echo "[$ts] BACKSTOP still-running $mark -- 재부착 필요"; break; fi
  sleep 110
done
echo "WATCHER_EXIT mark=$mark $(date '+%F %H:%M:%S')"
