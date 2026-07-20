# base seed48 — 6ep

## 요약
- **홀드아웃 0.7482** (2026-07-06 새벽, 치타 A5000 공유 GPU1, 총 ~4h)
- seed47(0.7481)과 사실상 동률 — 6ep 재현성 확인
- epoch별 점수/누적시간 (홀드아웃 학습 구간, epoch당 ~17분):

| ep | Macro-F1 | 누적 시간 |
|:--:|:---:|:---:|
| 1 | 0.6142 | 17분 |
| 2 | 0.6665 | 35분 |
| 3 | 0.7015 | 52분 |
| 4 | 0.7300 | 1h10m |
| 5 | 0.7416 | 1h28m |
| 6 | **0.7482** | 1h45m |

- 이후 전체 재학습(refit, 8754스텝) ~2h → zip 생성 새벽 3시경

## 구성
- 공통 레시피(README), 차이: 6ep / seed 48 / batch 16 × grad_accum 3 (공유 GPU 9GB 제약)
- 실행: 치타서버 `~/dacon/run48`, nohup, `EPOCHS=6 SEED=48 BATCH=16 GRAD_ACCUM=3`

## 산출물
- probs: `artifacts/hybrid/holdout_probs_s48.npy` + idx_s48 (**치타 분할** — s46/47과 동일, 구분할과는 교집합으로만 비교)
- 모델: `submits/submit_base2_s48.zip` (fp32; 블렌드 포장 시 fp16 변환)
- 블렌드 참여: v23 후보군 (2026-07-06 분석 예정)
