# base seed47 — 8ep, 단일모델 최고

## 요약
- **홀드아웃 0.7481 — 단일모델 신기록** (2026-07-05 밤, 치타 A5000 공유 ~5h)
- epoch별: 1ep 0.6035 → 2ep 0.6651 → 3ep 0.7021 → 4ep 0.7315 → 5ep 0.7376 → 6ep 0.7429 → 7ep(미포착) → **8ep 0.7481**
- epoch당 소요: ~17분(홀드아웃 구간)

## 구성
- 공통 레시피, 차이: 8ep / seed 47 / batch 16 × accum 3
- 실행: 치타서버 ~/dacon/run47

## 산출물
- probs: `artifacts/hybrid/holdout_probs_s47.npy` + idx_s47 (**치타 분할**)
- 모델: submits/submit_base2_s47.zip
- 블렌드 참여: **v22(0.7746) 현역**
