# v14 RoBERTa-large ([SRC] 없음) — 강하지만 서버에서 불리

## 요약
- **홀드아웃 0.7129** (2026-07-03, Kaggle T4 커밋 ~8h)
- 스태킹 기여: v15f(고전+large) LB 0.7207, v20(고전+s44+large) LB 0.7634
- **v21에서 제외됨**: [SRC] 없는 구형 직렬화라 서버 분포(au·초반스텝)에서 약함 — large를 빼자 LB가 오히려 상승(0.7701)

## 구성
- klue/roberta-large(337M), v12와 동일 직렬화([SRC] 없음, 384), 3ep, LR 1e-5, batch 16×accum 4, fp16 저장
- 스크립트: `open/scripts/colab_train_large.py`

## 산출물
- probs: `artifacts/hybrid/holdout_probs_L.npy` + idx_L (**Kaggle 분할**)
- 모델: submits/submit_v14.zip (fp16 backbone 643MB)

## 배포 교훈 (값비쌌던 것)
- **fp32 업캐스트 함정**: from_pretrained는 fp16 저장본도 fp32로 로드 → `net.half()` 없인 서버 T4에서 3배 느려져 3연속 시간초과(LB 0.654~0.670). half() 명시 후 정상(v15f 5분23초)
- 이 모델의 후계 = large_src_seed50 ([SRC]+512+5ep 재학습)
