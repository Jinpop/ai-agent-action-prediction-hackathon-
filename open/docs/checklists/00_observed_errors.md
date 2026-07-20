# 지금까지 관측된 오류 사전 (재발 금지)

1. **CI 복원 오류**: rolling-window 시작 CI를 none으로 가정 → 정확도 43.7%뿐인 train_mint_ci를 "깨끗한 복원"으로 오선언.
2. **마스킹 해석 오류**: 필드 제외가 아니라 0/False/빈값 변환이었음 — budget=0 ∧ elapsed=0이 완벽한 mint 도메인 표식이 됨.
3. **mint 수량 오해**: 6,859행은 독립 상태가 아니라 canonical 3,180개의 window 변형. real 중복 18개 제거 후 고유 text/action 최대치는 3,162개.
4. **복원 가능성 누락**: CI 3,108·dirty 3,016·open 2,910행은 복원 가능했는데 시도 없이 폐기. exact 교집합 2,863행.
5. **불가능 필드 혼입**: budget/elapsed는 정확 복원 불가 — 미래 source 값·zero sentinel 사용 금지.
6. **pretext 메타 누출**: 119d만 0화하고 transcript [META]는 남겨두고 text-only라고 오칭.
7. **pretext holdout 누수**: gate stage A에 holdout 세션 mint 620행 포함 → 홀드아웃 평가 무효.
8. **A/B 교란**: v54 blank@3,162 vs v55 keep@2,863을 metadata A/B로 해석 — 행 집합·처리 동시 상이.
9. **RNG 계보 오류**: 모델 생성 후 set_seed 호출; REFIT_ONLY를 기존 seed 모델의 정확 재현으로 오기술.
10. **manifest 과장**: 한 줄 요약을 완전한 환경 기록으로 호칭. 반대로 전체 printenv는 자격증명 유출 위험.
11. **산출물 계약 혼동**: hp_*.npy는 이름과 달리 raw logits인 파일 존재 — softmax 여부 sidecar로 명시.
12. **홀드아웃 오용**: 홀드아웃 순위로 모델 폐기·"최강" 호칭. 홀드아웃은 밴드 자격 필터일 뿐.
13. **워처 오류**: pgrep -f가 watcher 셸 자체를 세어 죽은 학습을 실행 중으로 판단. 디스크 full로 로그 없이 죽는 런 존재 → mtime 기준.
14. **상태변경 사고**: sleep 포함 원격 명령을 중단해도 kill/launch는 이미 실행됨 → 상태변경은 즉시반환형, 검증은 즉답형 별도.
15. **문서 계보 오류**: script docstring에 v47 계보·옛 멤버·w-probe 0.5 잔존(실코드 0.6).
16. **버전 불일치**: prep은 sklearn 1.9.0 생성, requirements는 1.6.1 — 임의 수정 금지, 경고 포함 로딩 smoke + 생성 버전 기록.
17. **해석 용어 오류**: 결정론적 서버 차이를 "노이즈"로 표현 금지 → "학습 draw·seed·처리 상호작용 범위".
18. **슬롯 기록 오류**: 제출 후에도 사용 슬롯 미갱신 — 제출 즉시 갱신.
