# zip 생성 전 체크

1. [ ] **diff 사전 선언**: v49 대비 예상 diff를 먼저 선언. 공통 멤버·고전·feat·requirements는 바이트 동일.
2. [ ] **script diff 최소**: 실행 diff는 원칙적으로 SEED_DIRS와 버전 설명(docstring)뿐.
3. [ ] **비복사본**: 후보 모델이 다른 모델의 복사본이 아닌지 backbone/head SHA 확인.
4. [ ] **SHA 체인**: 치타/H100 원본 pack → 로컬 staging → zip entry SHA 전부 대조.
5. [ ] **fp16·finite**: backbone은 fp16·finite, head는 finite.
6. [ ] **kf 계약**: params 185,290,752 / head 887→256→14 / meta 119 / max_len 768.
7. [ ] **RoBERTa 계약**: 대응 max_len(512)·head shape 별도 확인 (params 110,618,112 = s48과 동일).
8. [ ] **prep 계약**: prep["actions"] == feat.ACTIONS, scaler 열 순서·feature 수 일치.
9. [ ] **실행 코드 확인**: truncation_side="left", GPU에서 net.half().
10. [ ] **블렌드 계약**: 0.6×mean(neural)+0.4×classic, classic 0.45/0.40/0.15.
11. [ ] **순서 계약**: logits를 softmax 전에 평균하지 않았는지, class order 안 섞였는지.
12. [ ] **sklearn 불일치**: 오늘 임의 수정으로 신규 변수 만들지 말 것 — 경고 포함 로딩 smoke + 생성 버전 기록만.

자동화: `scratchpad/verify_zip.py`가 1~11을 검사한다. 12는 E2E 로그로 확인.
