#!/bin/bash
# ALL-INFO H100 체인: Stage A(pretext stable-meta 3ep) → Stage B(real REFIT_ONLY 6ep).
# 서버측 순차 실행(완료감시 최소화). 마커: STAGEA_DONE → STAGEB_DONE → ALLINFO_DONE.
# 레시피 불변(batch16×accum3·LR2e-5·GRAD_CKPT0·seed79). base=로컬 safetensors(CVE 우회).
set -e
cd ~/dacon
PY=~/dacon/env/bin/python
BASE=$HOME/dacon/kfdeberta_st
R=run_allinfo_s79
rm -rf $R; mkdir -p $R; cd $R
ln -sfn ../data data
cp ../colab_train_base2.py ../feat.py .

# ---- Stage A: stable-meta pretext (mint 6829, 3ep, text-only 숫자메타0화) ----
echo "STAGEA_START $(date +%FT%T)"
env CUDA_VISIBLE_DEVICES=0 MODEL_NAME=$BASE MAX_LEN=768 EPOCHS=3 BATCH=16 GRAD_ACCUM=3 \
    SEED=79 LR=2e-5 GRAD_CKPT=0 PRETEXT=1 PRETEXT_META=stable META_NUM_EXT=1 \
    TARGET_BALANCE=1 TB_V2=1 \
    EXTRA_DATA=data/train_mint_allinfo_stable.jsonl \
    $PY -u colab_train_base2.py > stageA.log 2>&1
[ -d pretext_backbone ] || { echo "STAGEA_FAIL: no pretext_backbone"; tail -20 stageA.log; exit 2; }
echo "STAGEA_DONE $(date +%FT%T)"

# ---- Stage B: real 70k REFIT_ONLY (6ep, 확장 transcript+125d, 홀드아웃 없음) ----
echo "STAGEB_START $(date +%FT%T)"
env CUDA_VISIBLE_DEVICES=0 MODEL_NAME=$BASE MAX_LEN=768 EPOCHS=6 BATCH=16 GRAD_ACCUM=3 \
    SEED=79 LR=2e-5 GRAD_CKPT=0 REFIT_ONLY=1 INIT_BACKBONE=$PWD/pretext_backbone \
    META_NUM_EXT=1 META_TRANS_EXT=1 EXTRA_DATA="" \
    $PY -u colab_train_base2.py > stageB.log 2>&1
[ -d model_sub ] || { echo "STAGEB_FAIL: no model_sub"; tail -20 stageB.log; exit 3; }
# 산출물 SHA (회수 대조용)
sha256sum model_sub/backbone/model.safetensors model_sub/head.pt model_sub/prep.pkl > model_sub/SHA256SUMS.txt 2>/dev/null || true
echo "STAGEB_DONE $(date +%FT%T)"
touch ALLINFO_DONE
echo "ALLINFO_DONE $(date +%FT%T)"
