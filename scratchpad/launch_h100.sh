#!/bin/zsh
# 엘리스 H100: frozen 코드/데이터 push → kf768 direct-mint 3런 병렬(seed 79v2/83/84)
# 규칙(07-12 승인): 단일 H100 3병렬(4금지)·레시피 불변(batch16×accum3·LR2e-5·6ep)·
#   frozen cb3ce5d8·산출물="H100 training draw"·완료 후 인스턴스 종료.
set -e
EOPT=(-i ~/.ssh/elice.pem -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes)
PORT=<ELICE_PORT>
EHOST=elicer@<ELICE_TUNNEL_HOST>
L=/Users/<USER>/Documents/Dacon_236694_AI_Agent
EXPECT_SHA=cb3ce5d8522b0d115a1ec49ffae074d8706477579cfd3769d3593360bcf8a998

# 1) frozen 코드 + 데이터 번들 push (기존 7/7 구버전 덮어쓰기)
scp -P $PORT $EOPT "$L/open/scripts/colab_train_base2.py" "$L/open/scripts/feat.py" \
    "$L/scratchpad/conv_fp16.py" "$L/scratchpad/h100_data.tgz" $EHOST:dacon/
echo "PUSH OK"

# 2) 데이터 추출 + sentencepiece 설치 + SHA/lib 검증 (미일치면 abort)
ssh -p $PORT $EOPT $EHOST 'set -e; cd ~/dacon
tar xzf h100_data.tgz
GOT=$(sha256sum colab_train_base2.py | cut -d" " -f1)
if [ "$GOT" != "cb3ce5d8522b0d115a1ec49ffae074d8706477579cfd3769d3593360bcf8a998" ]; then
  echo "FATAL: script SHA $GOT != frozen cb3ce5d8"; exit 3; fi
echo "SHA OK (frozen cb3ce5d8)"
~/dacon/env/bin/python -m pip -q install sentencepiece 2>&1 | tail -1 || true
~/dacon/env/bin/python -c "import torch,transformers,sklearn,sentencepiece,safetensors; print(\"libs: torch\",torch.__version__,\"tf\",transformers.__version__,\"sk\",sklearn.__version__)"
echo "=== data 행수 ==="; wc -l data/train.jsonl data/train_mint.jsonl'
echo "PREP OK"

# 3) 3런 발사 (단일 H100 GPU0 공유; kf768 batch16 ~21GB×3 ≈ 63GB < 80GB)
ssh -p $PORT $EOPT $EHOST 'set -e; cd ~/dacon
PY=~/dacon/env/bin/python
for S in 79 83 84; do
  R=run_h${S}; [ "$S" = "79" ] && R=run_h79v2
  rm -rf $R; mkdir -p $R; cd $R
  ln -sfn ../data data; cp ../colab_train_base2.py ../feat.py .
  echo "MODEL=kakaobank/kf-deberta-base MAX_LEN=768 EPOCHS=6 BATCH=16 GRAD_ACCUM=3 SEED=$S LR=2e-5 GRAD_CKPT=0 EXTRA_DATA=data/train_mint.jsonl launched=$(date +%FT%R) hw=H100_80GB pipeline=v2 note=H100_training_draw" > env_manifest.txt
  setsid env CUDA_VISIBLE_DEVICES=0 MODEL_NAME=kakaobank/kf-deberta-base MAX_LEN=768 EPOCHS=6 BATCH=16 GRAD_ACCUM=3 SEED=$S LR=2e-5 GRAD_CKPT=0 EXTRA_DATA=data/train_mint.jsonl nohup $PY -u colab_train_base2.py > train.log 2>&1 < /dev/null &
  disown; cd ..
done
echo "LAUNCHED $(ps aux | grep -c "[c]olab_train_base2") procs"'
echo "LAUNCH DONE"
