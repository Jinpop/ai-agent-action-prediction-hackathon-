#!/bin/bash
# ALL-INFO H100 발사기 (campaign all-info-h100-final-0715).
# ①env/GPU 점검 ②base(safetensors) 조건부 업로드 ③data/코드/체인 업로드 ④SHA 게이트 ⑤A→B 체인 nohup.
# 실제 발사(과금 학습 시작)는 fail-closed 12항 PASS + 사용자 SSH 확인 후에만 이 스크립트를 실행한다.
set -e
PEM=~/Downloads/<ELICE_KEY_ID>.pem
PORT=<ELICE_PORT>
EHOST=elicer@<ELICE_TUNNEL_HOST>
EOPT=(-i "$PEM" -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -o PubkeyAcceptedKeyTypes=+ssh-rsa -o HostKeyAlgorithms=+ssh-rsa)
L=/Users/<USER>/Documents/Dacon_236694_AI_Agent
FROZEN_SHA=78af69de4fe6383145da6b13f3b1686112d2c20f9f4fb80cfcda307b0ef0d0e9
BASE_SHA=fea650417514d6f22c06f8368ce3c4f928daf77eff4cad8ec49f799c5509adc8   # kfdeberta_st/model.safetensors

echo "===[1] env/GPU 점검==="
ssh -p $PORT "${EOPT[@]}" $EHOST 'mkdir -p ~/dacon; echo HOME_OK; \
  python -c "import torch,transformers; print(\"torch\",torch.__version__,\"tf\",transformers.__version__,\"cuda\",torch.cuda.is_available())" 2>&1 | tail -1; \
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 | head -1; \
  ls -d ~/dacon/env 2>/dev/null && echo HAS_VENV || echo NO_VENV'

echo "===[2] base(safetensors) 조건부 업로드==="
REMOTE_BASE_SHA=$(ssh -p $PORT "${EOPT[@]}" $EHOST 'sha256sum ~/dacon/kfdeberta_st/model.safetensors 2>/dev/null | cut -d" " -f1' || true)
if [ "$REMOTE_BASE_SHA" = "$BASE_SHA" ]; then
  echo "base 이미 존재(SHA 일치) — 업로드 스킵"
else
  echo "base 업로드 필요(remote=$REMOTE_BASE_SHA) — tar+scp $(du -sh $L/scratchpad/kfdeberta_st | cut -f1)"
  tar czf "$L/scratchpad/kfdeberta_st.tgz" -C "$L/scratchpad" kfdeberta_st
  scp -P $PORT "${EOPT[@]}" "$L/scratchpad/kfdeberta_st.tgz" $EHOST:dacon/
  ssh -p $PORT "${EOPT[@]}" $EHOST 'cd ~/dacon && tar xzf kfdeberta_st.tgz && rm -f kfdeberta_st.tgz && \
    GOT=$(sha256sum kfdeberta_st/model.safetensors | cut -d" " -f1); \
    [ "$GOT" = "'"$BASE_SHA"'" ] && echo "BASE SHA OK" || { echo "FATAL base SHA $GOT"; exit 4; }'
fi

echo "===[3] data/코드/체인 업로드==="
scp -P $PORT "${EOPT[@]}" \
  "$L/scratchpad/allinfo_frozen/colab_train_base2.py" \
  "$L/scratchpad/allinfo_frozen/feat.py" \
  "$L/scratchpad/conv_fp16.py" \
  "$L/scratchpad/allinfo_chain.sh" \
  "$L/scratchpad/h100_allinfo_data.tgz" $EHOST:dacon/

echo "===[4] SHA 게이트 + data 추출==="
ssh -p $PORT "${EOPT[@]}" $EHOST 'set -e; cd ~/dacon
GOT=$(sha256sum colab_train_base2.py | cut -d" " -f1)
[ "$GOT" = "'"$FROZEN_SHA"'" ] || { echo "FATAL colab SHA $GOT != frozen '"$FROZEN_SHA"'"; exit 3; }
echo "FROZEN SHA OK ('"$FROZEN_SHA"' 앞8=78af69de)"
tar xzf h100_allinfo_data.tgz
wc -l data/train.jsonl data/train_mint_allinfo_stable.jsonl
chmod +x allinfo_chain.sh'

echo "===[5] A→B 체인 발사(nohup)==="
ssh -p $PORT "${EOPT[@]}" $EHOST 'cd ~/dacon
setsid nohup bash allinfo_chain.sh > allinfo_chain.log 2>&1 < /dev/null &
disown; sleep 3
echo "PID: $(pgrep -f allinfo_chain.sh | head -1)"
echo "=== chain.log head ==="; head -5 allinfo_chain.log 2>/dev/null || echo "(로그 생성 대기)"'
echo "LAUNCH DONE — 완료감시(run-monitor) 부착 필요"
