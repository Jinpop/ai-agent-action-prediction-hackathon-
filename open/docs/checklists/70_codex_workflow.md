# 감사 게이트 운영 체크리스트 (2026-07-14 하네스 재설계)

Claude=실행자. **내부 교차검증**(Claude 서브에이전트)이 필수 게이트다. **외부 감사(Codex)** 는 사용자 명시 승인 시에만
병행한다. 상세는 `open/coordination/README.md` + `open/coordination/agent_contracts.md`.

## 최우선 (외우기)
- **기본 = PREPARE_ONLY.** Codex는 execute + **사용자 명시 승인**에서만 1회 기동. 자동 재시도 없음.
- **AUDIT_PASS ≠ 행동 승인.** PASS여도 발사·중단·kill·유료GPU·삭제·제출은 각각 별도 사용자 승인(`action_state=WAITING_USER`).
- **safe_next_action은 권고문**일 뿐 실행 권한이 아니다.

## 내부 교차검증 (필수 게이트, 전환점마다)
1. `context-curator`로 **최소 증거 패킷**을 만든다(SHA 고정). 감사자는 이 패킷 + 담당 체크리스트만 읽는다(전체 통독 금지).
2. 전환점별 역할 호출(README 표): plan_review→experiment-planner/data-lineage/training-contract/code-diff, pre_launch→split-leakage/training-contract, post_gate→metric/blend/split-leakage, pre_submit→zip-verifier, post_lb→lb-interpreter, 마무리→integration-auditor.
3. verdict: PASS(진행 자격)/BLOCK(수정 후 재검증)/NEEDS_HUMAN(중지·질문). 동일 lineage+finding 2회 BLOCK → NEEDS_HUMAN 승격.
4. **판정은 파일로 영속화**(`open/coordination/results/` 또는 crosscheck/, 스키마 준수). 구두 금지.

## 외부 감사 (Codex) — 사용자 명시 승인 시에만
1. **PREPARE_ONLY**: `external-audit-packager`(또는 지휘자)가 audit_request JSON 작성 → `run_codex_gate.py prepare <req.json>`.
   - 필수 필드(스키마 v3): `stage, objective, verify_items, downstream_action, proposed_next_action`.
     증거: `code_sha256`/`data_sha256`는 **실측 64-hex만**(설명·미측정은 `sha_notes`로 분리 — #5), `evidence_paths`.
   - 결과 gate_state=**WAITING_CODEX_APPROVAL**(또는 동일 audit_key면 cache hit로 재사용, 호출 없음). audit_key 확보.
2. **사용자에게 Codex 1회 호출 승인을 명시적으로 요청.** "발사해"급 직접 승인 없이는 여기서 멈춘다.
3. 승인 확인 후 **지휘자만** execute:
   ```
   run_codex_gate.py execute <prepared.json> --approve-request-id <rid> --approve-audit-key <audit_key>
   ```
   - **정확히 1회.** exit≠0/timeout/400이어도 자동 재시도 없음 → INFRA_ERROR 기록·사용자 보고. 재감사는 **새 request_id + 새 승인**.
4. 분기: 0=PASS(진행 자격) / 10=BLOCK(수정 후 새 request 재감사) / 20=NEEDS_HUMAN(중지·질문) / 30=INFRA_ERROR(진행 금지, 재감사는 새 승인).

## 안전 불변식
- Codex/교차검증자는 파일 무수정(러너가 before/after 매니페스트로 검증, 위반 시 INFRA_ERROR fail-closed).
- 경로는 repo 내부 regular file만(절대경로·`..`·symlink escape는 Codex 미호출 차단).
- 위험 행동(제출·유료GPU·삭제·원격kill·발사)은 게이트가 실행하지 않음 — 각각 사용자 승인.
- **현재 실행 중인 H100/치타 프로세스는 이 게이트로 중단·재시작·변경하지 않는다.**
- 저장 직후 `validate_audits.py` 1회(legacy는 schema_version+backfilled+manifest로 판정 — mtime 아님).
