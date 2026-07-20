#!/usr/bin/env python3
"""PreToolUse(Bash) 훅: 명령 패턴에 따라 해당 작업의 검증 체크리스트 경로를 주입.
07-12 사용자 규약 — 전 항목 통과한 결과만 '검증 완료'로 보고 가능."""
import json
import re
import sys

BASE = "open/docs/checklists"
RULES = [
    (r"colab_train_base2|EPOCHS=|PRETEXT|REFIT_ONLY|INIT_BACKBONE|nohup.*train",
     f"{BASE}/10_pre_training.md"),
    (r"[Oo][Oo][Ff]|distill|soft_target|teacher|증류",
     f"{BASE}/20_oof_distill.md"),
    (r"zip -r|verify_zip|stage_v|submit_v\d+\.zip",
     f"{BASE}/30_pre_zip.md, {BASE}/40_zip_self.md, {BASE}/50_per_candidate.md"),
    (r"실험로그|HANDOFF|슬롯",
     f"{BASE}/60_post_submit.md"),
]

d = json.load(sys.stdin)
cmd = (d.get("tool_input") or {}).get("command", "") or ""
hits = [path for pat, path in RULES if re.search(pat, cmd)]
if hits:
    msg = ("[checklist-guard] 이 작업 유형의 강제 체크리스트: " + " / ".join(dict.fromkeys(hits))
           + " — 전 항목 통과 없이는 '검증 완료'라고 보고하지 말 것(미통과 항목 명시).")
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "additionalContext": msg}}, ensure_ascii=False))
