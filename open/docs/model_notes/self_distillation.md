# 자기증류 (★성공)

## 레시피
OOF=1로 5-fold soft label 생성(시드별) → make_soft_target.py로 평균 → SOFT_TARGET=soft.npz DISTILL_T=2.0 DISTILL_ALPHA=0.5로 학습(loss=0.5CE+0.5KD).

## 결과 (H100, 2026-07-07)
- OOF 2seed(80,81) 각 5-fold, 4.78it/s 동시, ~2h → 교사(2seed평균) train전체 macroF1 **0.7513**
- **distill 학생 seed90 (T=2.0, α=0.5, 6ep): 홀드아웃 0.7585 (6ep eval 0.7609)**
  - epoch별: 0.624/0.700/0.736/0.750/0.758/0.761 — 6ep에도 상승 중(7ep 시도 여지)
  - 일반시드 최고(s51 0.7506) +0.008, 3-seed 앙상블(0.7555)도 단독 상회 — 대회 최대 단일 돌파
- 확장: 학생 91/92/93 풀런(refit포함) H100 동시발사 13:15, 완료 ~14:30 → v26 포장
- 함정 기록: npz lazy 반복접근 금지(수정됨), 원격 발사는 setsid+nohup+</dev/null

## 다음 판정 포인트
- 학생 앙상블 다양성(같은 교사 공유 → 상관 우려): 학생4 vs 학생2+시드2 혼합 비교
- **LB 이전 여부가 최종 판정** — v26 제출로 확인
