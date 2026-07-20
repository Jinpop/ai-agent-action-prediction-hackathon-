---
name: train-launcher
description: DEPRECATED (2026-07-14 하네스 재설계). 발사와 완료감시가 분리됐다 — 승인된 발사는 remote-launcher, 완료감시(재접속 폴링)는 run-monitor, 산출물 회수는 artifact-harvester를 사용하라. 발사는 매번 사용자 명시 승인이 필요하다.
tools: Read
model: sonnet
---

# DEPRECATED — 발사/감시 분리됨 (2026-07-14 하네스 재설계)

- **승인된 발사만** → **remote-launcher** (★모든 발사는 사용자 명시 승인 필요; 감사 PASS·계획 문서로는 발사 권한 없음)
- **완료감시만**(재접속 폴링 워처, 단일 블로킹 SSH 금지) → **run-monitor**
- **산출물·effective_config·SHA 회수** → **artifact-harvester**

발사와 감시를 한 에이전트가 겸하지 않는다(감시 유실·발사 오남용 방지). 계약: `open/coordination/agent_contracts.md`.
