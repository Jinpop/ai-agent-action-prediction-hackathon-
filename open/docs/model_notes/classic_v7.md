# 고전 3모델 (v7 계열) — 모든 블렌드의 고정 멤버

## 요약
- **홀드아웃 0.6089 / 단독 LB 0.6179** (v7, 2026-07-02)
- 학습: 로컬 macOS (open/.venv, sklearn 1.6.1), `open/scripts/train.py`
- 아티팩트: `open/artifacts/model/artifacts_clean.pkl` (115MB; HGB 학습용 RNG 제거판 — 서버 numpy 호환)

## 구성 (3모델 + 피처)
- 텍스트: word TF-IDF(1,2)gram 50k + char_wb(3,5) 40k → 합쳐 sparse ~90k차원
- **HGB**: TFIDF→TruncatedSVD(300) + 메타 119d = dense 419d, HistGradientBoosting(iter500, leaf63, early stop), sample_weight=balanced
- **LogReg**: sparse 텍스트 + MaxAbsScaler(메타) hstack ~90k차원, C=3, max_iter 2000, balanced
- **NB**: ComplementNB(alpha 0.3), 텍스트 sparse만
- 메타피처는 `open/scripts/feat.py`의 build_meta_row (요금제/워크스페이스/키워드/상태신호/history 통계)
- 블렌드 가중치(홀드아웃 튜닝): **HGB 0.45 / LogReg 0.40 / NB 0.15**

## 홀드아웃 확률 캐시
`open/artifacts/classic_holdout_probs.npz` (ph/pl/pn/classes) — **Kaggle 분할 기준**(holdout_idx.npy와 정렬), 80%로 재학습해 생성 (누수 없음)

## 역할과 한계
- 단독으론 ~0.61 천장 (탐색4형제 구분 불가)
- 블렌드에서 0.4 가중으로 참여 — 트랜스포머와 실수 패턴이 달라(respond_only 등 명시신호 클래스에 강함) 안정화 기여
- NB는 스태킹 기여 0으로 판명됐으나 블렌드 가중 0.15로 잔류 (제거해도 무해)
