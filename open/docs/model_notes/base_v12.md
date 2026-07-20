# v12 base 풀트랜스크립트 — 직렬화 혁명

## 요약
- **홀드아웃 0.7031 / 단독 LB 0.6921** (2026-07-03, Kaggle T4)
- 의의: history를 [U]/[A](args→결과) 대화록으로 직렬화 → 단일모델이 이전 스태킹 전체를 넘음
- [SRC] 프리픽스는 아직 없음(=au 미인지), MAX_LEN 384

## 구성
- 공통 레시피(README) 기준, 차이: [SRC] 없음 / MAX_LEN 384 / 3ep / batch 64 / seed 42
- 스크립트: `open/scripts/colab_train_hybrid2.py`
- **truncation_side="left" 필수** — 1차 시도는 기본(right) 잘림으로 [P]프롬프트가 39% 샘플에서 소실돼 0.5459 참사, 수정 후 0.70

## 산출물
- probs: `artifacts/hybrid/holdout_probs2.npy` + idx2 (**Kaggle 분할**)
- 모델: submits/submit_v12.zip (토크나이저 구버전 호환 패치 + requirements 고정 수리판)

## 교훈 (제출 사고 2건)
- Kaggle transformers 5.x가 저장한 tokenizer_config(TokenizersBackend)는 서버 4.x에서 못 읽음 → BertTokenizerFast로 패치
- 이후 모든 스크립트에 자동 패치 반영됨
