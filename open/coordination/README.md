# Claude 내부 교차검증 + Codex 외부 감사 게이트 (2026-07-14 하네스 재설계)

**Claude = 실행자.** 위험 전환점에서 **내부 교차검증**(Claude 서브에이전트)을 필수로 돌린다 — 이는 실행자·검증자가
모두 Claude 계열이라 **"독립 감사"가 아니라 "내부 교차검증"**이다. **외부 감사(Codex)** 는 오류 상관성을 줄이는 추가
장치이며, **사용자가 그 1회 호출을 명시 승인했을 때만** 실행한다.

## 최우선 정책 (충돌 시 이 절이 우선)
1. **기본 동작은 PREPARE_ONLY.** 요청·증거 묶음만 만들고 Codex 프로세스를 시작하지 않는다.
2. **외부 감사(Codex)는 사용자 명시 승인 1건당 정확히 1회.** 오류·timeout·400이어도 **자동 재시도 없음.** 재시도·force-reaudit는 **새 승인** 필요.
3. **★AUDIT_PASS는 행동 승인과 완전히 별개다.** 감사 PASS는 "감사 통과(진행 자격)"일 뿐 — **학습발사·중단·kill·유료GPU·삭제·제출은 각각 별도의 사용자 명시 승인** 없이는 `action_state=WAITING_USER`로 남는다. 어떤 stage의 PASS도 발사·제출 권한을 만들지 않는다.
4. **safe_next_action은 권고문일 뿐 실행 권한이 아니다.**
5. **실제 제출·유료 GPU 개통·파일 삭제·원격 kill·발사는 계속 사용자 승인 대상**(러너·감사자가 실행하지 않음).

## 상태기계
```
gate_state:  DRAFT → PREPARED → WAITING_CODEX_APPROVAL → RUNNING →
             PASS | BLOCK | NEEDS_HUMAN | INFRA_ERROR
action_state: WAITING_USER | AUTHORIZED | EXECUTED   (러너는 항상 WAITING_USER만 기록)
```
- **INFRA_ERROR는 verdict가 아니다.** 러너/인프라 실패(exit≠0·timeout·400·스키마부적합·경로탈출·승인누락·파일수정)는 가짜 NEEDS_HUMAN 감사로 변환하지 않고 `infra/`에 INFRA_ERROR 레코드로 기록한다(스키마 `infra_error.schema.json`).

## 역할 분리
1. **문서(HANDOFF·실험로그) 수정은 docs-ledger-writer만.** 실험로그는 EOF append-only.
2. **내부 교차검증 서브에이전트**(data-lineage/split-leakage/training-contract/code-diff/metric/blend/integration/zip-verifier)는 **read-only 판정만.** 파일 무수정.
3. **Codex 외부 감사**는 `--sandbox read-only --ephemeral`로만 실행(파일 무수정 — 매니페스트로 검증). 출력은 raw audit_result JSON 하나뿐.
4. Codex/교차검증 **BLOCK을 지휘자가 임의로 무시하지 않는다.** 동일 lineage+finding 연속 미해결 BLOCK 2회 → NEEDS_HUMAN 승격.

## 검증 체크포인트 (이 전환점에서만 — 미세작업마다 호출 금지)
| stage | 무엇을 감사하나 | 내부 교차검증 역할 |
|---|---|---|
| `plan_review` | 목적·직접 대조군·단일변수·교란·정보가치 | experiment-planner + data-lineage/training-contract/code-diff |
| `pre_launch` | 코드/data SHA·effective config·split·누수·ETA | split-leakage + training-contract |
| `anomaly_review` | 크래시·속도저하·환경차이·우회책 부작용 | code-diff + training-contract |
| `post_gate` | holdout 정렬·logit/prob 계약·누수·밴드 자격 | metric-auditor + blend-evaluator + split-leakage |
| `post_training` | refit 완주·pack↔staging↔zip SHA·manifest | artifact-harvester(회수) + code-diff |
| `pre_submit` | verify_zip 전항 + 5행 E2E | zip-verifier |
| `post_lb` | 직접 대조 delta·교란·챔피언·후속 해석 | lb-interpreter → docs-ledger-writer |
| (전 지점 마무리) | 상태전이·문서정합 최종 | integration-auditor |

내부 교차검증 PASS는 **진행 자격**이다. 부작용 단계(발사·kill·제출)는 여기에 더해 **사용자 명시 승인**을 받아야 실행된다.

## 외부 감사(Codex) 러너 — prepare / execute 분리
```
# 1) PREPARE_ONLY — 요청 검증·증거 SHA 실측·audit_key·프롬프트/prepared 번들. Codex 미호출.
open/.venv/bin/python scratchpad/run_codex_gate.py prepare <request.json>
#    → gate_state=WAITING_CODEX_APPROVAL (또는 동일 audit_key면 cache hit로 기존 결과 재사용, 호출 없음)

# 2) EXECUTE — ★사용자가 이 1회 호출을 명시 승인한 뒤 지휘자만. 정확히 1회, 자동 재시도 없음.
open/.venv/bin/python scratchpad/run_codex_gate.py execute <prepared.json> \
    --approve-request-id <rid> --approve-audit-key <audit_key>
#    → PASS|BLOCK|NEEDS_HUMAN (감사 결과) | INFRA_ERROR (러너/인프라 실패)
```
- 종료코드: 0=PASS 5=WAITING_CODEX_APPROVAL 10=BLOCK 20=NEEDS_HUMAN 30=INFRA_ERROR 40=CLI 사용오류.
- 승인은 **1호출 유효**(execute 후 `locks/<rid>.executed`로 소진). 재감사는 새 request_id + 새 승인.
- 준비(prepare)는 서브에이전트 **external-audit-packager**가 대행 가능(호출은 금지 — 지휘자만 execute).
- 모델/추론: gpt-5.5/xhigh(env `CODEX_GATE_MODEL`/`CODEX_GATE_REASONING`). 격리 테스트: `CODEX_GATE_BIN=scratchpad/fake_codex.sh` + `COORD_ROOT=<임시>`.

## 관측성·회계 (파일)
```
open/coordination/prepared/<rid>.json      # 승인 대기 번들(gate_state=WAITING_CODEX_APPROVAL)
open/coordination/prompts/<rid>.txt        # Codex 프롬프트
open/coordination/logs/<rid>/attempt-01.*  # ★attempt별 분리(덮어쓰기 없음): jsonl 원시이벤트 + stderr
open/coordination/results_raw/<rid>.json   # Codex 원시(raw) 출력
open/coordination/results/<rid>.json       # 러너-정규화 결과(request_id/model/시각/attempt/audit_key는 러너 기록)
open/coordination/infra/<rid>-<kind>.json  # INFRA_ERROR 레코드(verdict 아님)
open/coordination/ledger/usage.jsonl       # ★호출별 usage 원장(model/reasoning/시작·종료/cache/exit/token). 미제공 토큰=null(추정 금지)
open/coordination/cache/index.json         # audit_key → 결과(동일 증거 재감사 방지)
open/coordination/state/escalation.json    # lineage_id별 finding fingerprint 카운트(무관 실험 혼입 없음)
open/coordination/legacy_manifest.json     # v3 이전 pre-versioning 기록 동결(validate_audits가 legacy 판정에 사용 — mtime cutoff 대체)
```

## 동시성·안전
- request는 `.tmp` 작성 후 atomic rename로 게시. 기존 결과 재사용 금지(cache는 audit_key 기준 명시 재사용).
- request의 artifact SHA가 현재 파일 SHA와 다르면 **Codex 호출 없이 INFRA_ERROR(evidence_sha_mismatch)**.
- 경로는 realpath 기준 **repo 내부 regular file**만 허용 — 절대경로·`..` 탈출·symlink escape는 **Codex 미호출 차단**(path_escape).
- Codex 파일수정 감지(before/after 매니페스트) → **INFRA_ERROR(codex_modified_files) fail-closed**.
- 검증기: `open/.venv/bin/python scratchpad/validate_audits.py`(legacy는 schema_version+backfilled+manifest로만 판정 — mtime 아님). 격리 테스트: `open/.venv/bin/python scratchpad/test_codex_gate.py`.

## 서브에이전트 계약
역할·입출력·쓰기권한은 `open/coordination/agent_contracts.md`. 어떤 역할도 HANDOFF/실험로그 전체를 기본으로 읽지 않고, `context-curator` 최소 패킷 + 담당 체크리스트만 읽는다.

기존 zip 검증 체크리스트·`verify_zip.py`는 재사용(중복 구현 금지).
