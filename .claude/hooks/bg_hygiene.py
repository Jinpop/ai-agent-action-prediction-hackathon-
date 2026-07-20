#!/usr/bin/env python3
"""PreToolUse(Bash) 훅: run_in_background=true일 때 백그라운드 위생 체크리스트를
모델 컨텍스트에 주입 (비차단). CLAUDE.md 위생 규칙의 하네스 강제 장치 (07-12)."""
import json
import sys

d = json.load(sys.stdin)
ti = (d.get("tool_input") or {})
bg = ti.get("run_in_background")
cmd = ti.get("command") or ""
if bg:
    msg = ("[bg-hygiene hook] 새 백그라운드 작업 추가 전 필수 점검(CLAUDE.md 규칙): "
           "①기존 bg 중 목적 소멸·대체된 것 즉시 TaskStop ②워처 교체 시 구 워처 중지 "
           "③발사 ssh·검증 대기자 잔류 금지(검증은 sleep 없는 즉답형 foreground 1회) "
           "④동시 한도: 워처 1 + 연산 1 — 초과 상태면 지금 정리부터 할 것")
    # ★07-13: 원격 완료감시 워처면 워처 프로토콜도 주입 (단일 블로킹 SSH 사고 재발방지)
    if "ssh" in cmd and ("DONE" in cmd or "pgrep" in cmd or "while" in cmd or "for i in" in cmd):
        msg += (" | [완료감시 프로토콜] ★단일 장기 블로킹 SSH 금지 — 반드시 재접속 폴링"
                "(매 90~120s 짧은 새 SSH, 연결실패는 다음 폴서 재시도, 상한 ETA×2). "
                "상한 도달 시 'still running-재부착' 출력. ★ETA+여유까지 알림 없으면 워처 불신하고 "
                "즉답 foreground SSH로 직접 완주 확인(07-13 8h 유휴 사고 방지). 완주 즉시 수확→다음 GPU 배정.")
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "additionalContext": msg}}, ensure_ascii=False))
