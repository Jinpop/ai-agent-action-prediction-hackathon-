# 서브에이전트 역할·입출력·쓰기권한 계약 (2026-07-14 하네스 재설계)

메인 세션(Claude 지휘자)은 **사용자 소통·전략·승인 상태·최종 조립**만 담당한다. 아래 17개 역할을 별도
`.claude/agents/*.md`로 분리하고, 각자의 입력/출력 JSON 계약과 쓰기 권한을 고정한다.

## 최우선 정책 (모든 에이전트 공통)
- Claude 내부 감사는 **"내부 교차검증(internal cross-check)"** 이라 부른다 — "독립 감사" 아님(실행자·검증자 모두 Claude 계열이라 오류 상관성 잔존).
- **외부 감사(Codex)는 사용자가 그 1회 호출을 명시 승인했을 때만** 실행한다. 기본은 **PREPARE_ONLY**.
- **AUDIT_PASS ≠ 행동 승인.** 학습발사·중단·kill·유료GPU·삭제·제출은 각각 명시적 사용자 승인 없이는 `action_state=WAITING_USER` 유지.
- **safe_next_action은 권고문일 뿐 실행 권한이 아니다.**
- **★컨텍스트 위생(#13):** 어떤 역할도 HANDOFF/실험로그 **전체를 기본으로 읽지 않는다.** `context-curator`가 만든
  **SHA 고정 최소 패킷**(`open/coordination/packets/<id>.json`)과 **담당 체크리스트만** 읽는다.
- 문서(HANDOFF·실험로그) 수정은 **docs-ledger-writer 한 명만.** 실험로그는 반드시 **EOF append-only**.

## 공통 입력/출력 뼈대
- 입력(감사·평가·계획 역할): `{ "packet_path": "open/coordination/packets/<id>.json", "checklist_path": "open/docs/checklists/<n>.md", "task": "<한 줄>" }`
- 출력(교차검증 역할): `cross_check_result`
  `{ "role": "...", "verdict": "PASS|BLOCK|NEEDS_HUMAN", "findings": [{"id","summary","evidence(file:line)","severity"}],
     "verified_claims": [{"claim","evidence"}], "unverified_claims": [...], "confounders": [...], "auditor_model": "..." }`
  — verdict는 판정일 뿐, 파일 수정·발사·제출 권한이 아니다. 지휘자가 `results/`(또는 `crosscheck/`)에 영속화.

## 역할·쓰기권한 표
| 에이전트 | 한 줄 역할 | tools | 쓰기 권한 |
|---|---|---|---|
| context-curator | 최신 상태에서 역할별 **최소 증거 패킷**(SHA 고정) 생성 | Read,Grep,Bash | read-only(패킷 JSON 반환; 지휘자가 packets/에 기록) |
| experiment-planner | 대조군·변경변수·정보가치 설계 | Read,Grep | read-only(plan JSON 반환) |
| data-lineage-auditor | 데이터 출처·dedup·복원 가능성 교차검증 | Read,Bash,Grep | read-only |
| split-leakage-auditor | hidx·세션 누수·OOF 정렬 교차검증 | Read,Bash,Grep | read-only |
| training-contract-auditor | stage별 epoch/LR/batch/env matched 여부 | Read,Bash,Grep | read-only |
| code-diff-auditor | 직접 대조 코드 diff·미선언 변경 | Read,Bash,Grep | read-only |
| remote-launcher | **승인된** 발사만 수행 | Bash,Read | 원격 실행(사용자 승인 발사만); 로컬 read-only |
| run-monitor | 발사와 분리해 **완료감시만** | Bash,Read | read-only(재접속 폴링) |
| artifact-harvester | 모델·로그·config·SHA 회수 | Bash,Read | harvest 디렉터리(logs/·scratchpad)만 기록 |
| metric-auditor | F1·class order·logit/prob 재계산 | Read,Bash | read-only |
| blend-evaluator | **밴드 자격만** 판정(우열·챔피언 판정 금지) | Read,Bash,Grep | read-only |
| package-builder | **승인된 구성의** staging/zip 생성 | Bash,Read,Write | 승인 구성 staging/zip만 기록 |
| zip-verifier | zip **read-only** 검증 | Bash,Read | read-only |
| lb-interpreter | 서버 delta·confounder·챔피언 해석 | Read,Grep | read-only(해석 JSON 반환) |
| docs-ledger-writer | **HANDOFF 갱신 + 실험로그 EOF append 전담** | Read,Edit,Write | ★유일한 문서 writer |
| external-audit-packager | **Codex용 최소 증거 묶음만 생성**(호출 금지 — prepare까지만) | Read,Bash,Write | 번들/prepared만 기록; execute·Codex 절대 금지 |
| integration-auditor | 전체 상태 전이·문서 정합 최종 확인 | Read,Bash,Grep | read-only |

## 게이트 러너 연동 (외부 감사)
- `external-audit-packager`는 `run_codex_gate.py prepare <req.json>`까지만 수행(=PREPARE_ONLY, Codex 미호출).
- 실제 `execute`(Codex 1회 호출)는 **사용자 명시 승인 문구**를 받은 뒤 **지휘자만** 실행한다. 승인은 1회 유효·자동 재시도 없음.
- 승인 정보: `--approve-request-id <rid> --approve-audit-key <key>`. AUDIT_PASS가 나와도 발사/제출은 별도 사용자 승인.

## 승격 규율
- 내부 교차검증에서 **동일 lineage_id + finding fingerprint** 연속 미해결 BLOCK 2회 → NEEDS_HUMAN 승격(무관 실험 혼입 없음).
- 감사자 판정은 파일로 영속화(구두 금지). 지휘자만 파일 수정, 감사자는 read-only.
