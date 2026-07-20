# OOF·증류 체크

1. [ ] **커버리지**: OOF ID가 대상 전체를 중복·누락 없이 정확히 한 번씩 덮는지 확인.
2. [ ] **세션 분리**: 같은 세션의 real/mint가 train과 held 양쪽에 갈라지지 않아야 함.
3. [ ] **정렬**: OOF74·OOF79·고전 teacher를 ID와 feat.ACTIONS 순서로 정렬.
4. [ ] **softmax 선행**: raw logits에는 모델별 softmax 적용 후 teacher 결합 (hp/oof 파일은 raw logits일 수 있음 — sidecar 확인).
5. [ ] **matched temperature**: KD temperature는 teacher·student 양쪽 동일 적용(v26 버그 = teacher T=1 vs student T=2). 확률합 1·finite 검사.
6. [ ] **in-sample 금지**: in-sample teacher prediction을 OOF라고 사용하지 않음.
7. [ ] **슬롯 계약**: 증류 student는 사전 지정한 한 슬롯만 교체, 배포 head 구조 불변.

재료 위치: 고전 OOF = `open/artifacts/oof/oof_classic_probs.npy`(+ids, probs·ACTIONS순),
kf OOF74 = `scratchpad/oof_logits_seed74.npy`(raw logits, ids는 치타 run_oof74/oof_ids_seed74.json),
kf OOF79 = 치타 run_oof79 (완료 시 npy+ids 회수).
