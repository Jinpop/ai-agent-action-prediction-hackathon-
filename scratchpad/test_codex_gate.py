#!/usr/bin/env python3
"""Codex 게이트 격리 테스트 (2026-07-14 재설계). ★fake Codex + 격리 COORD_ROOT만 사용.
실제 Codex·SSH·학습·zip·제출은 절대 건드리지 않는다.

사용: open/.venv/bin/python scratchpad/test_codex_gate.py [code|docs|all]
  code(기본): 러너/스키마/캐시/승인/재시도/승격 격리 테스트
  docs:       문서 정합(상충 문구 제거·정책 명문화) grep 검사
종료코드: 0=전부 PASS, 1=하나라도 FAIL
"""
import subprocess, os, json, sys, hashlib, shutil, glob, datetime

REPO = "/Users/<USER>/Documents/Dacon_236694_AI_Agent"
PY = os.path.join(REPO, "open/.venv/bin/python")
GATE = os.path.join(REPO, "scratchpad/run_codex_gate.py")
FAKE = os.path.join(REPO, "scratchpad/fake_codex.sh")
SCHEMA_DIR = os.path.join(REPO, "open/coordination/schemas")
TESTROOT = os.path.join(REPO, "scratchpad/gate_test")
COORD = os.path.join(TESTROOT, "coord")
CALLOG = os.path.join(TESTROOT, "callog")
REQDIR = os.path.join(TESTROOT, "reqs")
INJECTED = os.path.join(REPO, "scratchpad/_codex_added_test.py")

RESULTS = []


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def base_env(**extra):
    e = dict(os.environ)
    e.update({
        "COORD_ROOT": COORD, "CODEX_GATE_BIN": FAKE, "CODEX_GATE_SCHEMA_DIR": SCHEMA_DIR,
        "CODEX_GATE_POLL": "0.2", "FAKE_CALL_LOG": CALLOG, "CODEX_GATE_REPO": REPO,
    })
    e.update(extra)
    return e


def run(subcmd, extra_env=None):
    env = base_env(**(extra_env or {}))
    p = subprocess.run([PY, GATE] + subcmd, capture_output=True, text=True, env=env, cwd=REPO)
    out = None
    try:
        out = json.loads(p.stdout)
    except Exception:
        out = {"_raw": p.stdout, "_stderr": p.stderr}
    return p.returncode, out


def calls():
    if not os.path.exists(CALLOG):
        return 0
    return sum(1 for _ in open(CALLOG))


def reset_all():
    shutil.rmtree(TESTROOT, ignore_errors=True)
    os.makedirs(REQDIR, exist_ok=True)
    if os.path.exists(INJECTED):
        os.remove(INJECTED)


def write_req(rid, stage, verify_items, evidence, downstream="launch", lineage=None, extra=None):
    req = {
        "schema_version": "3", "request_id": rid, "created_at": "2026-07-14T12:00:00Z",
        "stage": stage, "policy_version": "2026-07-14", "objective": f"test {rid}",
        "verify_items": verify_items, "downstream_action": downstream,
        "proposed_next_action": "test next",
        "code_sha256": {p: sha(os.path.join(REPO, p)) for p in evidence},
    }
    if lineage:
        req["lineage_id"] = lineage
    if extra:
        req.update(extra)
    path = os.path.join(REQDIR, rid + ".json")
    json.dump(req, open(path, "w"), ensure_ascii=False, indent=2)
    return path


def prepared_path(rid):
    return os.path.join(COORD, "prepared", rid + ".json")


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


# ---------------- CODE TESTS ----------------

def t01_prepare_only_no_call():
    reset_all()
    rid = "plan_review-20260714T120000Z-t1prep"
    req = write_req(rid, "plan_review", ["item-a"], ["AGENTS.md"])
    ec, out = run(["prepare", req])
    check("T01 prepare exit=WAITING_CODEX_APPROVAL(5)", ec == 5, f"exit={ec}")
    check("T01 gate_state WAITING_CODEX_APPROVAL", out.get("gate_state") == "WAITING_CODEX_APPROVAL")
    check("T01 action_state WAITING_USER", out.get("record", {}).get("action_state") == "WAITING_USER")
    check("T01 subprocess 호출 0", calls() == 0, f"calls={calls()}")
    check("T01 usage ledger 미기록(0줄)", not os.path.exists(os.path.join(COORD, "ledger", "usage.jsonl")))
    check("T01 실행 로그 디렉터리 미생성", not os.path.isdir(os.path.join(COORD, "logs", rid)))
    check("T01 prepared 번들 존재", os.path.exists(prepared_path(rid)))


def t02_reject_bad_before_call():
    reset_all()
    # wrong type: verify_items 문자열
    p = os.path.join(REQDIR, "r.json")
    def w(obj):
        json.dump(obj, open(p, "w"))
        return p
    base = lambda: {"schema_version": "3", "request_id": "plan_review-20260714T120000Z-bad1",
                    "created_at": "2026-07-14T12:00:00Z", "stage": "plan_review",
                    "policy_version": "2026-07-14", "objective": "x", "verify_items": ["a"],
                    "downstream_action": "none", "proposed_next_action": "y"}
    b = base(); b["verify_items"] = "notarray"
    ec, out = run(["prepare", w(b)])
    check("T02a wrong-type 거부(exit 30)", ec == 30 and out.get("record", {}).get("error_kind") == "request_schema_invalid", f"exit={ec}")
    n0 = calls()
    # extra field
    b = base(); b["request_id"] = "plan_review-20260714T120000Z-bad2"; b["bogus_field"] = 1
    ec, out = run(["prepare", w(b)])
    check("T02b 추가필드 거부(exit 30)", ec == 30 and out.get("record", {}).get("error_kind") == "request_schema_invalid", f"exit={ec}")
    # bad SHA (non-hex)
    b = base(); b["request_id"] = "plan_review-20260714T120000Z-bad3"; b["code_sha256"] = {"AGENTS.md": "not-a-real-sha"}
    ec, out = run(["prepare", w(b)])
    check("T02c 비정상 SHA 거부(exit 30)", ec == 30, f"exit={ec} kind={out.get('record',{}).get('error_kind')}")
    # path traversal (..)
    b = base(); b["request_id"] = "plan_review-20260714T120000Z-bad4"; b["artifact_paths"] = ["../../../../etc/passwd"]
    ec, out = run(["prepare", w(b)])
    check("T02d ../ path traversal 거부(path_escape)", ec == 30 and out.get("record", {}).get("error_kind") == "path_escape", f"exit={ec} kind={out.get('record',{}).get('error_kind')}")
    # absolute escape
    b = base(); b["request_id"] = "plan_review-20260714T120000Z-bad5"; b["evidence_paths"] = ["/etc/hosts"]
    ec, out = run(["prepare", w(b)])
    check("T02e 절대경로 escape 거부(path_escape)", ec == 30 and out.get("record", {}).get("error_kind") == "path_escape", f"exit={ec} kind={out.get('record',{}).get('error_kind')}")
    check("T02 어떤 거부에서도 subprocess 0", calls() == 0, f"calls={calls()}")


def t03_execute_without_approval():
    reset_all()
    rid = "pre_launch-20260714T120000Z-t3noappr"
    req = write_req(rid, "pre_launch", ["s"], ["AGENTS.md"])
    run(["prepare", req])
    before = calls()
    ec, out = run(["execute", prepared_path(rid)])  # 승인 플래그 없음
    check("T03 승인없는 execute 거부(exit 30)", ec == 30, f"exit={ec}")
    check("T03 error_kind approval_missing", out.get("record", {}).get("error_kind") == "approval_missing")
    check("T03 subprocess 미기동", calls() == before, f"delta={calls()-before}")


def t04_one_approval_one_call():
    reset_all()
    rid = "pre_launch-20260714T120000Z-t4pass"
    req = write_req(rid, "pre_launch", ["s"], ["AGENTS.md"])
    _, pout = run(["prepare", req])
    ak = pout["audit_key"]
    before = calls()
    ec, out = run(["execute", prepared_path(rid), "--approve-request-id", rid, "--approve-audit-key", ak],
                  {"FAKE_SCENARIO": "pass"})
    check("T04 PASS(exit 0)", ec == 0 and out.get("gate_state") == "PASS", f"exit={ec} gs={out.get('gate_state')}")
    check("T04 subprocess 정확히 1회", calls() - before == 1, f"delta={calls()-before}")
    rec = out["record"]
    check("T04 verdict_source=codex", rec.get("verdict_source") == "codex")
    check("T04 결과 results/에 영속", os.path.exists(os.path.join(COORD, "results", rid + ".json")))
    check("T04 raw가 results_raw/에 분리 저장", os.path.exists(os.path.join(COORD, "results_raw", rid + ".json")))
    check("T04 attempt별 로그 분리(attempt-01)", os.path.exists(os.path.join(COORD, "logs", rid, "attempt-01.jsonl")))
    led = os.path.join(COORD, "ledger", "usage.jsonl")
    lines = [json.loads(l) for l in open(led)] if os.path.exists(led) else []
    codex_calls = [l for l in lines if l["event"] == "codex_call"]
    check("T04 usage ledger codex_call 1건", len(codex_calls) == 1, f"n={len(codex_calls)}")
    check("T04 ledger tokens=null(추정 안 함)", codex_calls and codex_calls[0]["tokens"] == {"input": None, "output": None, "total": None})
    return rid, out


def t05_no_auto_retry():
    for scen, kind in [("exit1", "codex_exit_nonzero"), ("http400", "codex_http_error"), ("timeout", "codex_timeout")]:
        reset_all()
        rid = f"pre_launch-20260714T120000Z-t5{scen.replace('4','4x')}"
        # rid slug must be [a-z0-9-]{4,}
        rid = f"pre_launch-20260714T120000Z-t5-{scen}"
        req = write_req(rid, "pre_launch", ["s"], ["AGENTS.md"])
        _, pout = run(["prepare", req])
        ak = pout["audit_key"]
        before = calls()
        env = {"FAKE_SCENARIO": scen}
        if scen == "timeout":
            env.update({"CODEX_GATE_TIMEOUT": "1", "FAKE_SLEEP": "5"})
        ec, out = run(["execute", prepared_path(rid), "--approve-request-id", rid, "--approve-audit-key", ak], env)
        check(f"T05 {scen}: INFRA_ERROR(exit 30)", ec == 30 and out.get("gate_state") == "INFRA_ERROR", f"exit={ec}")
        check(f"T05 {scen}: error_kind={kind}", out.get("record", {}).get("error_kind") == kind, f"got={out.get('record',{}).get('error_kind')}")
        check(f"T05 {scen}: 자동 재시도 없음(정확히 1회 기동)", calls() - before == 1, f"delta={calls()-before}")
        check(f"T05 {scen}: PASS로 미승격(results/ 없음)", not os.path.exists(os.path.join(COORD, "results", rid + ".json")))


def t06_cache_hit():
    reset_all()
    # 1st: 실제 PASS 감사 → 캐시 저장
    rid1 = "post_gate-20260714T120000Z-t6first"
    req1 = write_req(rid1, "post_gate", ["cache-item"], ["AGENTS.md"])
    _, p1 = run(["prepare", req1]); ak1 = p1["audit_key"]
    run(["execute", prepared_path(rid1), "--approve-request-id", rid1, "--approve-audit-key", ak1], {"FAKE_SCENARIO": "pass"})
    before = calls()
    # 2nd: 다른 rid, 동일 stage+verify_items+policy+evidence → 동일 audit_key → cache hit
    rid2 = "post_gate-20260714T120000Z-t6second"
    req2 = write_req(rid2, "post_gate", ["cache-item"], ["AGENTS.md"])
    ec, out = run(["prepare", req2])
    check("T06 동일 audit_key", out.get("audit_key") == ak1, f"{out.get('audit_key')} vs {ak1}")
    check("T06 cache hit(PASS, exit 0)", ec == 0 and out.get("cache_hit") is True and out.get("gate_state") == "PASS", f"exit={ec} hit={out.get('cache_hit')}")
    check("T06 cache hit는 subprocess 미기동", calls() - before == 0, f"delta={calls()-before}")
    led = [json.loads(l) for l in open(os.path.join(COORD, "ledger", "usage.jsonl"))]
    check("T06 ledger에 cache_hit 이벤트(호출 아님)", any(l["event"] == "cache_hit" for l in led))


def t07_force_reaudit_needs_approval():
    reset_all()
    rid1 = "post_gate-20260714T120000Z-t7first"
    req1 = write_req(rid1, "post_gate", ["fr-item"], ["AGENTS.md"])
    _, p1 = run(["prepare", req1]); ak1 = p1["audit_key"]
    run(["execute", prepared_path(rid1), "--approve-request-id", rid1, "--approve-audit-key", ak1], {"FAKE_SCENARIO": "pass"})
    before = calls()
    # force-reaudit prepare(캐시 우회) → 다시 승인 대기
    rid2 = "post_gate-20260714T120000Z-t7force"
    req2 = write_req(rid2, "post_gate", ["fr-item"], ["AGENTS.md"])
    ec, out = run(["prepare", req2, "--force-reaudit"])
    check("T07 force-reaudit prepare는 캐시 우회→승인대기", ec == 5 and out.get("cache_hit") is False, f"exit={ec} hit={out.get('cache_hit')}")
    # 승인 없이 execute --force-reaudit → 거부
    ec, out = run(["execute", prepared_path(rid2), "--force-reaudit"])
    check("T07 force-reaudit라도 승인없으면 execute 거부", ec == 30 and out.get("record", {}).get("error_kind") == "approval_missing", f"exit={ec}")
    check("T07 force-reaudit 거부 시 subprocess 0", calls() - before == 0, f"delta={calls()-before}")


def t08_pass_launch_still_waiting_user(rid_out):
    rid, out = rid_out
    rec = out["record"]
    check("T08 PASS의 safe_next_action='launch'", rec.get("safe_next_action") == "launch", f"got={rec.get('safe_next_action')}")
    check("T08 그래도 action_state=WAITING_USER(행동 승인 분리)", rec.get("action_state") == "WAITING_USER", f"got={rec.get('action_state')}")


def t09_invalid_raw_not_promoted():
    reset_all()
    rid = "pre_submit-20260714T120000Z-t9inval"
    req = write_req(rid, "pre_submit", ["s"], ["AGENTS.md"], downstream="submit")
    _, pout = run(["prepare", req]); ak = pout["audit_key"]
    before = calls()
    ec, out = run(["execute", prepared_path(rid), "--approve-request-id", rid, "--approve-audit-key", ak], {"FAKE_SCENARIO": "invalid"})
    check("T09 invalid raw → INFRA_ERROR(exit 30)", ec == 30 and out.get("gate_state") == "INFRA_ERROR", f"exit={ec}")
    check("T09 error_kind result_schema_invalid", out.get("record", {}).get("error_kind") == "result_schema_invalid")
    check("T09 최종 PASS로 미승격(results/ 없음)", not os.path.exists(os.path.join(COORD, "results", rid + ".json")))
    check("T09 그래도 1회 기동(재시도 없음)", calls() - before == 1, f"delta={calls()-before}")


def t10_escalation_isolated_by_lineage():
    reset_all()
    def block(rid, lineage, evidence, verify):
        req = write_req(rid, "plan_review", verify, evidence, lineage=lineage)
        _, p = run(["prepare", req]); ak = p["audit_key"]
        return run(["execute", prepared_path(rid), "--approve-request-id", rid, "--approve-audit-key", ak],
                   {"FAKE_SCENARIO": "block", "FAKE_FID": "shared_fid"})
    # A 첫 BLOCK
    _, a1 = block("plan_review-20260714T120000Z-t10a1", "linA", ["AGENTS.md"], ["va"])
    check("T10 A1 BLOCK(미승격)", a1.get("gate_state") == "BLOCK" and a1.get("record", {}).get("escalated") is False)
    # B 첫 BLOCK (같은 finding id, 다른 lineage) — A에 영향 없어야
    _, b1 = block("plan_review-20260714T120000Z-t10b1", "linB", ["CLAUDE.md"], ["vb"])
    check("T10 B1 BLOCK(미승격, 무관 lineage)", b1.get("gate_state") == "BLOCK" and b1.get("record", {}).get("escalated") is False)
    # A 둘째 BLOCK (같은 lineage+finding) → 승격
    _, a2 = block("plan_review-20260714T120000Z-t10a2", "linA", ["open/coordination/README.md"], ["va"])
    check("T10 A2 같은 lineage+finding 2회→NEEDS_HUMAN 승격", a2.get("gate_state") == "NEEDS_HUMAN" and a2.get("record", {}).get("escalated") is True, f"gs={a2.get('gate_state')}")
    esc = json.load(open(os.path.join(COORD, "state", "escalation.json")))
    check("T10 escalation 카운터가 lineage별 분리(linA=2, linB=1)",
          esc.get("linA", {}).get("shared_fid") == 2 and esc.get("linB", {}).get("shared_fid") == 1, f"esc={esc}")


def t11_modify_scenario():
    reset_all()
    rid = "anomaly_review-20260714T120000Z-t11mod"
    req = write_req(rid, "anomaly_review", ["s"], ["AGENTS.md"], downstream="none")
    _, pout = run(["prepare", req]); ak = pout["audit_key"]
    before = calls()
    ec, out = run(["execute", prepared_path(rid), "--approve-request-id", rid, "--approve-audit-key", ak], {"FAKE_SCENARIO": "modify"})
    injected_now = os.path.exists(INJECTED)
    if injected_now:
        os.remove(INJECTED)  # 테스트 잔여물 정리
    check("T11 modify(파일수정) → INFRA_ERROR(exit 30)", ec == 30 and out.get("gate_state") == "INFRA_ERROR", f"exit={ec}")
    check("T11 error_kind codex_modified_files", out.get("record", {}).get("error_kind") == "codex_modified_files")
    check("T11 modified/added에 주입파일 기록", "scratchpad/_codex_added_test.py" in (out.get("record", {}).get("added_files", []) + out.get("record", {}).get("modified_files", [])))
    check("T11 verdict로 미승격(results/ 없음)", not os.path.exists(os.path.join(COORD, "results", rid + ".json")))
    check("T11 1회 기동", calls() - before == 1, f"delta={calls()-before}")


def t12_validate_catches_new_violation():
    reset_all()
    os.makedirs(os.path.join(COORD, "results"), exist_ok=True)
    os.makedirs(os.path.join(COORD, "requests"), exist_ok=True)
    # v3 결과인데 required 필드 누락 → 즉시 위반
    bad = {"schema_version": "3", "request_id": "plan_review-20260714T120000Z-t12bad", "verdict": "PASS"}
    json.dump(bad, open(os.path.join(COORD, "results", bad["request_id"] + ".json"), "w"))
    env = base_env()
    p = subprocess.run([PY, os.path.join(REPO, "scratchpad/validate_audits.py")], capture_output=True, text=True, env=env)
    check("T12 신규 v3 위반을 validate_audits가 exit 1로 잡음", p.returncode == 1, f"exit={p.returncode}")
    # schema_version 미표기 + manifest 밖 → 위반
    reset_all(); os.makedirs(os.path.join(COORD, "results"), exist_ok=True)
    nover = {"request_id": "plan_review-20260714T120000Z-t12nov", "verdict": "PASS"}
    json.dump(nover, open(os.path.join(COORD, "results", nover["request_id"] + ".json"), "w"))
    p = subprocess.run([PY, os.path.join(REPO, "scratchpad/validate_audits.py")], capture_output=True, text=True, env=env)
    check("T12 schema_version 미표기 신규도 exit 1", p.returncode == 1, f"exit={p.returncode}")
    # valid v3 결과는 통과 — 실제 게이트가 만든 결과로 검증
    reset_all()
    rid = "pre_launch-20260714T120000Z-t12ok"
    req = write_req(rid, "pre_launch", ["s"], ["AGENTS.md"])
    _, pout = run(["prepare", req]); ak = pout["audit_key"]
    run(["execute", prepared_path(rid), "--approve-request-id", rid, "--approve-audit-key", ak], {"FAKE_SCENARIO": "pass"})
    p = subprocess.run([PY, os.path.join(REPO, "scratchpad/validate_audits.py")], capture_output=True, text=True, env=env)
    check("T12 게이트가 만든 정상 v3 결과는 통과(exit 0)", p.returncode == 0, f"exit={p.returncode}\n{p.stdout[-400:]}")


def code_tests():
    print("\n== CODE TESTS (격리 COORD + fake Codex) ==")
    t01_prepare_only_no_call()
    t02_reject_bad_before_call()
    t03_execute_without_approval()
    ridout = t04_one_approval_one_call()
    t05_no_auto_retry()
    t06_cache_hit()
    t07_force_reaudit_needs_approval()
    t08_pass_launch_still_waiting_user(ridout)
    t09_invalid_raw_not_promoted()
    t10_escalation_isolated_by_lineage()
    t11_modify_scenario()
    t12_validate_catches_new_violation()


# ---------------- DOC TESTS ----------------

def _read(p):
    return open(os.path.join(REPO, p), encoding="utf-8").read()


def doc_tests():
    print("\n== DOC CONSISTENCY TESTS ==")
    readme = _read("open/coordination/README.md")
    check("D01 README에 'PASS → …자동 진행' 상충 문구 제거",
          "자동 진행" not in readme or "AUDIT_PASS" in readme,
          "여전히 '자동 진행' 존재")
    # 핵심 정책 명문화(다섯 문서 어딘가에)
    corpus = readme + _read("CLAUDE.md") + _read("AGENTS.md") + _read("open/docs/checklists/70_codex_workflow.md") + _read(".claude/agents/external-audit-packager.md" if os.path.exists(os.path.join(REPO, ".claude/agents/external-audit-packager.md")) else "open/coordination/README.md")
    check("D02 PREPARE_ONLY 기본 정책 명문화", "PREPARE_ONLY" in corpus)
    check("D03 '승인 1회 1호출/자동 재시도 없음' 명문화", ("1회" in corpus and "재시도" in corpus))
    check("D04 'AUDIT_PASS는 행동 승인과 별개' 명문화", "행동 승인" in corpus)
    check("D05 '내부 교차검증' 용어 도입(독립 감사 대체)", "내부 교차검증" in corpus)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "code"
    if mode in ("code", "all"):
        code_tests()
    if mode in ("docs", "all"):
        doc_tests()
    # 테스트 잔여물 완전 정리(격리 COORD·주입파일 삭제, 재생성 안 함)
    shutil.rmtree(TESTROOT, ignore_errors=True)
    if os.path.exists(INJECTED):
        os.remove(INJECTED)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n=== {npass} PASS / {nfail} FAIL / {len(RESULTS)} total ===")
    if nfail:
        print("FAIL 목록:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name} — {detail}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
