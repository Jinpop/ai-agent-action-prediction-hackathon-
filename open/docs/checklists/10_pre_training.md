# 학습 전 체크 (발사 직전 전 항목 확인)

1. [ ] **사전 등록**: 실험 목적·직접 대조군·변경 변수·제출 시 교체 슬롯을 한 줄로 실험로그에 사전 등록.
2. [ ] **SHA 저장**: frozen code·feat.py·JSONL·labels SHA256 저장 (effective_config.json이 자동 기록하는지 확인).
3. [ ] **effective config**: 전체 printenv가 아니라 코드가 읽은 effective config를 기본값 포함 JSON으로 저장.
4. [ ] **RNG 계보**: set_seed(SEED)가 모델 생성 전에 실행되는지 확인 (pipeline-v2).
5. [ ] **H100 동일 레시피**: H100에서도 batch16×accum3·LR·epoch 유지. 하드웨어 때문에 batch를 키우면 별도 학습법이 됨.
6. [ ] **H100 표기**: H100 산출물은 A5000 seed의 정확 재현이 아니라 **H100 training draw**로 기록.
7. [ ] **direct mint 계보**: gate 56,000 real + 2,559 mint / full refit 70,000 + 3,180 확인.
8. [ ] **pretext 행수 assert**: strict=3,162 / exact=2,863 / gate exact=2,314.
9. [ ] **pretext gate 누수 0**: gate stage A에 holdout 세션 mint 0행.
10. [ ] **blank/omit/keep 구분**: blank=빈 [META] 유지, omit=줄 자체 삭제, keep=정확 tier/CI/dirty/turn/open만 포함(budget/elapsed 0건).
11. [ ] **stage B real-only**: EXTRA_DATA 없음을 로그와 effective config 양쪽에서 확인.
12. [ ] **상호배타 assert**: PRETEXT ∧ MASK_MINT_META 동시 활성화 차단 확인.
13. [ ] **ETA 명시 + 완료감시 부착** (run_in_background 또는 원격 블로킹 워처, mtime 기준).
