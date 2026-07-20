#!/bin/bash
# ALL-INFO A100 인스턴스 환경 부트스트랩 (fresh 인스턴스: venv + 검증 스택 설치).
# 검증된 학습 스택: torch 2.5.1+cu121 · transformers 4.57.6 (기존 H100/치타와 동일 계열).
# safetensors base(kfdeberta_st) 사용이라 torch<2.6 CVE 무관. 완료 시 SETUP_DONE 마커.
set -e
mkdir -p ~/dacon
cd ~/dacon
echo "SETUP_START $(date +%FT%T)"
sudo apt-get update -qq 2>&1 | tail -1 || true
sudo apt-get install -y -qq python3.10-venv 2>&1 | tail -1 || true
python3 -m venv env
env/bin/pip install -q --upgrade pip 2>&1 | tail -1
env/bin/pip install -q "numpy<2" 2>&1 | tail -1
echo "== torch 설치(대용량, 수분) =="
env/bin/pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
echo "== ML 스택 =="
env/bin/pip install -q transformers==4.57.6 scikit-learn pandas datasets joblib accelerate safetensors sentencepiece 2>&1 | tail -2
env/bin/python -c "import torch,transformers,sklearn,safetensors,datasets,numpy,pandas,joblib; print('LIBS torch',torch.__version__,'tf',transformers.__version__,'np',numpy.__version__,'sk',sklearn.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOGPU')"
echo "SETUP_DONE $(date +%FT%T)"
touch ~/dacon/SETUP_DONE
