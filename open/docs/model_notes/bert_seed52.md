# bert-base seed52 — 백본 다양성 실험 (학습 중)

## 상태: 2026-07-06 오전 치타 GPU0에서 시작
## 가설
지금 앙상블은 전부 klue/roberta-base 형제라 실수 패턴이 유사(예측일치 91%).
**klue/bert-base**(다른 사전학습 모델)는 다른 실수를 할 것 → 블렌드 다양성 +.
성공 기준: 단독 0.73+ 이더라도 블렌드에서 +0.002 이상 기여하면 채택.

## 구성: 공통 레시피, MODEL_NAME=klue/bert-base / 6ep / seed 52 / batch16×accum3
## 완료 후 기입: epoch별 점수/시간, 블렌드 편입 여부
