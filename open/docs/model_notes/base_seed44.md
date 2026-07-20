# base seed44 — [SRC]+512 표준 레시피 확립 + 5ep

## 요약
- **홀드아웃 0.7440** (2026-07-04 새벽, Kaggle P100 배치커널 ~9h)
- epoch별: 1ep 0.6243 → 2ep 0.7024 → 3ep 0.7202(=base2 s43과 비교점) → 4ep 0.7404 → **5ep 0.7440**
- 의의: ① [SRC] au|sim 프리픽스 도입(au F1 0.51→0.77 반전의 시작) ② MAX_LEN 512 ③ "3ep는 수렴 전, 5ep+"를 실증
- epoch당 소요: ~55분 (P100, batch 48; eval 포함)

## 구성 (README 공통 레시피의 원조)
- klue/roberta-base, [SRC]+512, 5ep, LR 2e-5, batch 48(accum 없음), seed 44
- 실행: Kaggle 배치커널 (P100 배정 — 프리인스톨 torch가 sm_60 미지원이라 cu118 강제교체 런처 필요했음)

## 산출물
- probs: `artifacts/hybrid/holdout_probs_s44.npy` + idx_s44 (**Kaggle 분할**)
- 모델: submits/submit_base2_s44.zip... 주의: 실제 파일명은 초기 명명 문제로 submits/에 s44 zip이 없음 — v20/v21 스테이지에서 fp16 변환본만 사용됨. 원본 fp32는 Kaggle 커널(jinpop/dacon-base3-seed44-ep5) Output에 영구 보존
- 블렌드 참여: v20(LB 0.7634), v21c(0.7728). v22부터는 최약체로 제외

## 재현
Kaggle에서: `EPOCHS=5 SEED=44 BATCH=48 python colab_train_base2.py` (12h 제한상 P100은 5ep가 안전 최대)
