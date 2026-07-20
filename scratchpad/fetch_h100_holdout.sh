#!/bin/zsh
# H100 홀드아웃 로짓 회수 → 로컬 밴드체크(band_check_sanity.py) 즉시화.
# 홀드아웃 단계 완료(~step 7326) 후 각 run에 holdout_probs3.npy/holdout_idx3.npy 생성됨.
# usage: fetch_h100_holdout.sh  (3런 전부 시도, 없으면 스킵)
SCP=(-i ~/.ssh/elice.pem -P <ELICE_PORT> -o StrictHostKeyChecking=no -o BatchMode=yes)
SSH=(-i ~/.ssh/elice.pem -p <ELICE_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes)
EHOST=elicer@<ELICE_TUNNEL_HOST>
L=/Users/<USER>/Documents/Dacon_236694_AI_Agent/scratchpad
V=/Users/<USER>/Documents/Dacon_236694_AI_Agent/open/.venv/bin/python
cd /Users/<USER>/Documents/Dacon_236694_AI_Agent
# 어떤 런이 홀드아웃 산출물 준비됐는지 확인
ready=$(ssh $SSH $EHOST 'cd ~/dacon; for R in run_h79v2 run_h83 run_h84; do [ -f $R/holdout_probs3.npy ] && [ -f $R/holdout_idx3.npy ] && echo $R; done' 2>/dev/null)
echo "홀드아웃 준비된 런: ${ready:-없음}"
for R in ${(f)ready}; do
  suf=${R#run_}
  scp $SCP $EHOST:dacon/$R/holdout_probs3.npy "$L/hv_${suf}_probs.npy" 2>/dev/null
  scp $SCP $EHOST:dacon/$R/holdout_idx3.npy  "$L/hv_${suf}_idx.npy"   2>/dev/null
  echo "=== 밴드체크: $suf ==="
  $V scratchpad/band_check_sanity.py "$L/hv_${suf}_probs.npy" "$L/hv_${suf}_idx.npy" "$suf" 2>&1 | grep -vE "warnings.warn|NotOpenSSL|urllib3" | grep -E "SANITY|baseline|solo|candidate|delta|PASS|FAIL|ABORT"
done
