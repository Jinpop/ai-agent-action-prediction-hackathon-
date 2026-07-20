#!/usr/bin/env python3
"""Codex 외부 감사 게이트 CLI (2026-07-14 재설계). 코어는 codex_gate_lib.py.

★기본 동작은 PREPARE_ONLY. Codex 프로세스는 execute + 명시 승인에서만 1회 기동.

사용:
  run_codex_gate.py prepare <request.json> [--force-reaudit]
      요청 검증·증거 SHA 실측·audit_key·프롬프트/prepared 번들 생성. Codex 미호출.
      cache hit면 기존 결과 재사용(호출 없음).
  run_codex_gate.py execute <prepared.json> --approve-request-id <rid> --approve-audit-key <key> [--force-reaudit]
      ★사용자가 이 1회 호출을 명시 승인했을 때만 Codex를 정확히 1회 실행. 자동 재시도 없음.
  run_codex_gate.py <request.json>
      (하위호환) prepare와 동일 — PREPARE_ONLY.
  run_codex_gate.py status <request_id>

종료코드: 0=PASS  5=WAITING_CODEX_APPROVAL(prepared)  10=BLOCK  20=NEEDS_HUMAN
          30=INFRA_ERROR  40=CLI 사용오류
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codex_gate_lib as G


def _emit(out):
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _usage(msg):
    _emit({"gate_state": "GATE_USAGE", "message": msg})
    sys.exit(G.EXIT["GATE_USAGE"])


def cmd_prepare(argv):
    if not argv:
        _usage("usage: prepare <request.json> [--force-reaudit]")
    force = "--force-reaudit" in argv
    path = [a for a in argv if not a.startswith("--")][0]
    req = json.load(open(path))
    r = G.prepare(req, force_reaudit=force)
    _emit({"gate_state": r["gate_state"], "cache_hit": r.get("cache_hit"),
           "audit_key": r.get("audit_key"), "request_id": r["record"].get("request_id"),
           "action_state": r["record"].get("action_state", "WAITING_USER"),
           "record": r["record"]})
    sys.exit(r["exit_code"])


def cmd_execute(argv):
    pos = [a for a in argv if not a.startswith("--")]
    if not pos:
        _usage("usage: execute <prepared.json> --approve-request-id <rid> --approve-audit-key <key>")
    bundle = json.load(open(pos[0]))
    force = "--force-reaudit" in argv

    def flag(name):
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                return argv[i + 1]
        return None

    appr_rid = flag("--approve-request-id")
    appr_key = flag("--approve-audit-key")
    approval = None
    if appr_rid and appr_key:
        approval = {"approved": True, "request_id": appr_rid, "audit_key": appr_key}
    r = G.execute(bundle, approval, force_reaudit=force)
    rec = r["record"]
    _emit({"gate_state": r["gate_state"], "request_id": rec.get("request_id"),
           "action_state": rec.get("action_state", "WAITING_USER"),
           "error_kind": rec.get("error_kind"),
           "escalated": rec.get("escalated"),
           "safe_next_action": rec.get("safe_next_action"),
           "note": "★AUDIT_PASS는 행동 승인과 별개다. 학습발사·kill·삭제·제출은 action_state=WAITING_USER — 별도 사용자 승인 필요.",
           "record": rec})
    sys.exit(r["exit_code"])


def cmd_status(argv):
    if not argv:
        _usage("usage: status <request_id>")
    rid = argv[0]
    out = {"request_id": rid}
    for kind in ("prepared", "results", "infra"):
        for p in [G.d(kind, rid + ".json")] + (
                [G.d(kind, f) for f in (os.listdir(G.d(kind)) if os.path.isdir(G.d(kind)) else [])
                 if f.startswith(rid + "-")]):
            if os.path.exists(p):
                out[kind] = json.load(open(p))
    _emit(out)
    sys.exit(0)


def main():
    if len(sys.argv) < 2:
        _usage("usage: prepare|execute|status ... (기본 PREPARE_ONLY)")
    cmd = sys.argv[1]
    if cmd == "prepare":
        cmd_prepare(sys.argv[2:])
    elif cmd == "execute":
        cmd_execute(sys.argv[2:])
    elif cmd == "status":
        cmd_status(sys.argv[2:])
    elif os.path.isfile(cmd):
        # 하위호환: run_codex_gate.py <request.json> → PREPARE_ONLY
        cmd_prepare([cmd] + sys.argv[2:])
    else:
        _usage(f"알 수 없는 명령/파일: {cmd}")


if __name__ == "__main__":
    main()
