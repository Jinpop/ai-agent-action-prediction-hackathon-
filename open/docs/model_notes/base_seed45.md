# base seed45 — 6ep

## 요약
- **홀드아웃 0.7467** (2026-07-04, Kaggle P100 배치커널 ~10.5h)
- epoch별: 1ep 0.6278 → 2ep 0.6604 → 3ep 0.7190 → 4ep 0.7334 → 5ep 0.7426 → **6ep 0.7467**
- epoch당 소요: ~55분 (P100 batch 48)
- seed44와 예측 일치율 91.2% → 8.8% 다양성이 앙상블 가치

## 구성
- 공통 레시피, 차이: 6ep / seed 45
- 실행: Kaggle 배치커널 jinpop/dacon-seed45-ep6 (cu118 자가치유 런처)

## 산출물
- probs: `artifacts/hybrid/holdout_probs_s45.npy` + idx_s45 (**Kaggle 분할**)
- 모델: submits/submit_base2_s45.zip (fp32 443MB; v21/v22에서 fp16 213MB 변환 사용)
- 블렌드 참여: v21c(0.7728), **v22(0.7746) 현역**
