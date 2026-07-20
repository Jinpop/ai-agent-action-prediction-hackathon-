---
name: log-analyst
description: DEPRECATED (2026-07-14 하네스 재설계). 홀드아웃 밴드 자격 판정은 blend-evaluator로, F1·class order·logit/prob 재계산은 metric-auditor로 분리됐다. 새 작업에는 이 에이전트 대신 blend-evaluator/metric-auditor를 사용하라.
tools: Read
model: sonnet
---

# DEPRECATED — 분리됨 (2026-07-14 하네스 재설계)

- **밴드 자격만 판정**(홀드아웃 F1 → 밴드 통과 여부) → **blend-evaluator**
- **F1·class order·logit/prob 독립 재계산** → **metric-auditor**

홀드아웃은 제출 자격 필터일 뿐 우열 판단자가 아니다 — 우열·챔피언·제출선택은 LB·전략의 몫. 계약: `open/coordination/agent_contracts.md`.
