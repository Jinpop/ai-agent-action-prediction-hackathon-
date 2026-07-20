# 검증 체크리스트 (07-12 사용자 규약 — 강제)

**규칙: 아래 작업을 할 때는 반드시 해당 파일만 읽고 전 항목을 점검한다.
모든 항목을 통과한 결과만 "검증 완료"로 보고할 수 있다.
하나라도 미통과·미확인이면 "검증 완료"라는 표현을 쓰지 말고 미통과 항목을 명시한다.**

| 작업 | 읽을 파일 | 자동화 |
|---|---|---|
| 학습 발사 전 | `10_pre_training.md` | effective_config.json (colab_train_base2.py 내장) |
| OOF 생성·증류 | `20_oof_distill.md` | — |
| zip 생성 전 | `30_pre_zip.md` | `scratchpad/verify_zip.py` |
| zip 생성 후 | `40_zip_self.md` | `scratchpad/verify_zip.py` + 5행 E2E |
| 후보별 대조 확인 | `50_per_candidate.md` | — |
| 제출 결과 기록 | `60_post_submit.md` | — |
| 실수 재발 점검(수시) | `00_observed_errors.md` | — |

- 자동화 러너: `open/.venv/bin/python scratchpad/verify_zip.py <zip> <staging> <ref v49> <신멤버dir> <구멤버dir>`
- 5행 E2E: staging 클론에 `open/data/test.jsonl`+`sample_submission.csv` 넣고 script.py 실행 →
  "seed참여 평균 3.00개/행, 고전단독 0행" + rows=5 확인.
- `.claude/hooks/checklist_guard.py`가 Bash 명령 패턴에 따라 해당 파일 경로를 자동 주입한다.
