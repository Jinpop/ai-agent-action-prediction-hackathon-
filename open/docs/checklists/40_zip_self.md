# zip 자체 체크 (생성 후·제출 전)

1. [ ] **크기** < 1,000,000,000 bytes (decimal).
2. [ ] **CRC**·중복 경로·`../`·절대경로 검사.
3. [ ] **금지 엔트리 0**: data/, output/, __MACOSX, .DS_Store, checkpoint, memo, __pycache__/*.pyc.
4. [ ] **전수 SHA**: zip 내부 모든 파일이 staging과 SHA(CRC32) 일치.
5. [ ] **5행 E2E**: classic과 신경망 3개가 각각 5/5 참여.
6. [ ] **참여 지표**: 평균 seed 참여 3.00, fallback(고전단독) 0, 빈 action 0, 14클래스 외 action 0.
7. [ ] **시간 리스크 명시**: v49 동형 구조면 T4 9분44초급 — 16초 마진 위험을 브리핑에 명시.

자동화: 1~4 = `scratchpad/verify_zip.py`, 5~6 = 5행 E2E(클론 + open/data/test.jsonl), 7 = 브리핑 문구.
