# base seed46 — 7ep, 치타서버 1호

## 요약
- **홀드아웃 0.7449** (2026-07-05 저녁, 치타 A5000 공유 ~4h)
- epoch별: 1ep 0.5983 → 2ep 0.6783 → 3ep 0.7055 → 4ep 0.7229 → 5ep 0.7382 → 6ep(로그 미포착) → **7ep 0.7449**
- epoch당 소요: ~17분(홀드아웃 학습 구간, A5000 이웃공유+batch16×3) — P100(~55분) 대비 3배+
- 참고: 7ep(0.7449) < s45 6ep(0.7467) — epoch 단조증가 아님(seed 노이즈 수준)

## 구성
- 공통 레시피, 차이: 7ep / seed 46 / **batch 16 × grad_accum 3** (공유 GPU 8.5GB 제약)
- 실행: 치타서버 ~/dacon/run46, nohup (1차 시도는 batch 64로 이웃 프로세스와 충돌해 OOM — 감축 후 성공)

## 산출물
- probs: `artifacts/hybrid/holdout_probs_s46.npy` + idx_s46 (**⚠️ 치타 분할 — 자체 idx로만 채점**)
- 모델: submits/submit_base2_s46.zip
- 블렌드 참여: **v22(0.7746) 현역**
