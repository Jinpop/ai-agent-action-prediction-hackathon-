#!/bin/zsh
# H100 통과 런 회수: refit model_sub → fp16 변환(박스) → 로컬 pack 회수.
# usage: recover_h100.sh <runsuffix>  (예: h83 → run_h83, pack_h83)
# 이후 로컬에서 build_fallback.py로 s48+k79+<member> zip 포장.
set -e
SUF=${1:?usage: recover_h100.sh <runsuffix>}
EOPT=(-i ~/.ssh/elice.pem -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes)
PORT=<ELICE_PORT>; EHOST=elicer@<ELICE_TUNNEL_HOST>
L=/Users/<USER>/Documents/Dacon_236694_AI_Agent/scratchpad
R=run_$SUF; [ "$SUF" = "h79v2" ] && R=run_h79v2

ssh -p $PORT $EOPT $EHOST "set -e; cd ~/dacon
[ -d $R/model_sub ] || { echo 'FATAL: $R/model_sub 없음(refit 미완)'; exit 3; }
rm -rf pack_$SUF; mkdir -p pack_$SUF
cp -r $R/model_sub pack_$SUF/
~/dacon/env/bin/python conv_fp16.py $SUF
du -sh pack_$SUF/model_sub/backbone
python3 -c \"import os;sz=os.path.getsize('pack_$SUF/model_sub/backbone/model.safetensors');print('fp16 backbone MB',round(sz/1e6));assert 300e6<sz<500e6,'크기 이상'\""
scp -P $PORT $EOPT -r $EHOST:dacon/pack_$SUF "$L/pack_$SUF"
echo "RECOVERED pack_$SUF → $L/pack_$SUF"
