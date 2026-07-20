# v10 하이브리드 (1세대) — CLS+메타 결합의 원형

## 요약
- **홀드아웃 0.6519 / LB 0.6334** (2026-07-02, Colab T4)
- 의의: "순수 텍스트 트랜스포머(0.50)는 진다 → 메타피처를 CLS에 concat해야 이긴다"를 확립
- 현재는 후속 세대(v12+)에 대체돼 블렌드에서 제외 (v19 실험에서 추가 이득 0)

## 구성
- klue/roberta-base + HybridNet(CLS 768 ⊕ 메타 119 → MLP)
- 직렬화(구형): `[STATE] turn/last/result/ci/open/dirty [/STATE]` 프리픽스 + feat.build_text(유저발화 3개+행동이름 8개+열린파일 토큰+프롬프트)
- MAX_LEN 256, 3ep, batch 32, LR 2e-5, sqrt 가중
- 스크립트: `open/scripts/colab_train_hybrid.py`

## 산출물
- probs: `artifacts/hybrid/holdout_probs.npy` + `holdout_idx.npy` (**Kaggle/Colab 분할**)
- 모델: submits/submit_v10.zip 내 model_sub (fp32)

## 교훈
- history를 "행동 이름 토큰"으로 뭉개면 args/결과/순서 정보를 잃는다 → v12에서 풀 트랜스크립트로 진화
