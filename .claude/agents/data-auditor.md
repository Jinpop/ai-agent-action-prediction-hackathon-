---
name: data-auditor
description: DEPRECATED (2026-07-14 하네스 재설계로 세분화됨). "독립 감사" 단일 에이전트는 폐지되고, 내부 교차검증이 전문 역할로 분리됐다. 데이터 출처는 data-lineage-auditor, split/누수는 split-leakage-auditor, 학습계약은 training-contract-auditor, 코드 diff는 code-diff-auditor, metric 재계산은 metric-auditor, 밴드 자격은 blend-evaluator, 전체 정합은 integration-auditor, zip은 zip-verifier를 사용하라. 외부 감사(Codex)는 external-audit-packager로 준비하고 사용자 명시 승인 후 지휘자만 호출한다.
tools: Read
model: sonnet
---

# DEPRECATED — 세분화됨 (2026-07-14 하네스 재설계)

이 에이전트는 더 이상 필수 게이트가 아니다. 정책 변경 요약:
- Claude 내부 검토는 **"독립 감사"가 아니라 "내부 교차검증"** 이라 부른다(실행자·검증자 모두 Claude 계열이라 오류 상관성 잔존).
- 단일 data-auditor 대신 **전문 역할**로 분리한다(계약: `open/coordination/agent_contracts.md`):
  - 데이터 출처·dedup·복원 → **data-lineage-auditor**
  - hidx·세션 누수·OOF 정렬 → **split-leakage-auditor**
  - stage별 epoch/LR/batch/env → **training-contract-auditor**
  - 코드 diff·미선언 변경 → **code-diff-auditor**
  - F1·class order·logit/prob 재계산 → **metric-auditor**
  - 밴드 자격 → **blend-evaluator**
  - 전체 상태전이·문서 정합 → **integration-auditor**
  - zip 계약 → **zip-verifier**
- **외부 감사(Codex)** 는 지정 전환점에서 **사용자 명시 승인 시에만**. 준비는 **external-audit-packager**(PREPARE_ONLY), 실제 1회 호출은 지휘자가 `run_codex_gate.py execute`로. AUDIT_PASS는 발사·제출 등 행동 승인과 별개다.

새 작업에는 이 에이전트를 쓰지 말고 위 전문 역할을 호출하라.
