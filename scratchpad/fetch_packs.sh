#!/bin/zsh
# 치타: k81 최우선 변환·회수 → oof74 npy → H100 데이터 번들 → fallback 3팩(pt74c/pt78/k77)
set -e
SSH_OPTS=(-i ~/.ssh/cheetah.pem -p <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15)
SCP_OPTS=(-i ~/.ssh/cheetah.pem -P <CHEETAH_PORT> -o StrictHostKeyChecking=no -o ConnectTimeout=15)
HOST=<REMOTE_USER>@<CHEETAH_IP>
LOCAL=/Users/<USER>/Documents/Dacon_236694_AI_Agent/scratchpad

# 1) k81 변환 (fp16 완료본 있으면 스킵)
ssh $SSH_OPTS $HOST 'set -e; cd ~/dacon
SZ=$(stat -c %s pack_kf768_81/model_sub/backbone/model.safetensors 2>/dev/null || echo 0)
if [ "$SZ" -lt 100000000 ] || [ "$SZ" -gt 500000000 ]; then
  rm -rf pack_kf768_81; mkdir -p pack_kf768_81
  cp -r run_kf768_81/model_sub pack_kf768_81/
  ~/dacon/env/bin/python conv_fp16.py kf768_81
fi
du -sh pack_kf768_81'
scp $SCP_OPTS -r -q $HOST:dacon/pack_kf768_81 "$LOCAL/pack_kf768_81"
echo "K81 OK"

# 2) oof74 npy
scp $SCP_OPTS -q $HOST:dacon/run_oof74/oof_logits_seed74.npy "$LOCAL/oof_logits_seed74.npy"
echo "OOF74 OK"

# 3) H100 데이터 번들 (챔피언 레시피 입력 = run_kf768_79/data)
ssh $SSH_OPTS $HOST 'set -e; cd ~/dacon; du -sh run_kf768_79/data; tar -czf h100_data.tgz -C run_kf768_79 data'
scp $SCP_OPTS -q $HOST:dacon/h100_data.tgz "$LOCAL/h100_data.tgz"
echo "BUNDLE OK"

# 4) fallback 3팩 (fp16 완료본 있으면 스킵)
ssh $SSH_OPTS $HOST 'set -e; cd ~/dacon
for x in kf768pt74c kf768pt78; do
  SZ=$(stat -c %s pack_$x/model_sub/backbone/model.safetensors 2>/dev/null || echo 0)
  if [ "$SZ" -lt 100000000 ] || [ "$SZ" -gt 500000000 ]; then
    rm -rf pack_$x; mkdir -p pack_$x
    cp -r run_$x/model_sub pack_$x/
    ~/dacon/env/bin/python conv_fp16.py $x
  fi
done
SZ=$(stat -c %s pack_kf768_77/model_sub/backbone/model.safetensors 2>/dev/null || echo 0)
if [ "$SZ" -lt 100000000 ] || [ "$SZ" -gt 500000000 ]; then
  rm -rf pack_kf768_77; mkdir -p pack_kf768_77
  unzip -o -q run_kf768_77/submit_base2.zip "model_sub/*" -d pack_kf768_77/
  ~/dacon/env/bin/python conv_fp16.py kf768_77
fi
du -sh pack_kf768pt74c pack_kf768pt78 pack_kf768_77
df -h /home/<REMOTE_USER> | tail -1'
for x in kf768pt74c kf768pt78 kf768_77; do
  scp $SCP_OPTS -r -q $HOST:dacon/pack_$x "$LOCAL/pack_$x"
  echo "scp pack_$x OK"
done
echo ALLDONE
