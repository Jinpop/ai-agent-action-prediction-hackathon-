# [SRC]-large seed50 — 5ep (완료)

## 요약
- **홀드아웃 0.7450** (2026-07-06 오전, 치타 A5000 공유 GPU0, 총 ~11h)
- epoch별 (epoch당 ~57분, grad-ckpt+batch8×6):

| ep | Macro-F1 | 누적 |
|:--:|:---:|:---:|
| 1 | 0.6357 | 58분 |
| 2 | 0.6999 | 1h56m |
| 3 | 0.7249 | 2h53m |
| 4 | 0.7358 | ~3h50m |
| 5 | **0.7450** | ~4h48m |
- refit(7295스텝) ~6h → zip 10:12

## 가설 검증 결과
- 단독으론 base seed들과 동률(0.745) — [SRC]가 large를 살렸지만 base 대비 우위는 없음
- **다양성 기여 +0.0006** (치타 14k, 4seed 평균 대비): 미미. 아키텍처 차이 < seed 차이
- v23b(고전+s48+L50)로 LB 검증 예정 — 실패 시 large 계열은 완전 종료

## 구성
- klue/roberta-large, 공통 레시피 + 5ep / seed 50 / LR 1e-5 / batch 8×accum 6 / gradient checkpointing
- 실행: 치타 ~/dacon/runL

## 산출물
- probs: artifacts/hybrid/holdout_probs_L50.npy + idx_L50 (**치타 분할**)
- 모델: submits/submit_largeSRC_s50.zip (fp32 1.1GB; fp16 변환 644MB)
