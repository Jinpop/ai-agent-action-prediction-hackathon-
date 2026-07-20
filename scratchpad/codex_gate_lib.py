#!/usr/bin/env python3
"""Codex 외부 감사 게이트 코어 라이브러리 (2026-07-14 하네스 재설계).

정책(최우선):
- Claude 내부 감사는 "내부 교차검증"(internal cross-check)이라 부른다 — "독립 감사" 아님.
- Codex 외부 감사는 **사용자가 그 호출을 명시 승인했을 때만** 실행한다.
- 기본 동작은 **PREPARE_ONLY**: 요청·증거 묶음만 만들고 Codex 프로세스를 시작하지 않는다.
- Codex 호출 승인은 **1회 호출에만** 유효. 오류·timeout·400이어도 **자동 재시도 없음**.
- 재시도·force-reaudit에는 **새 사용자 승인** 필요.
- **AUDIT_PASS는 행동 승인과 완전히 별개**(action_state 축). 학습발사·중단·kill·유료GPU·삭제·제출은
  각각 명시적 사용자 승인 없이는 action_state=WAITING_USER 유지.
- safe_next_action은 **권고문일 뿐 실행 권한이 아니다**.

상태기계:
  gate_state:  DRAFT → PREPARED → WAITING_CODEX_APPROVAL → RUNNING →
               PASS | BLOCK | NEEDS_HUMAN | INFRA_ERROR
  action_state: WAITING_USER | AUTHORIZED | EXECUTED  (러너는 항상 WAITING_USER만 기록)

이 파일은 라이브러리다. CLI는 run_codex_gate.py. 실제 Codex는 CODEX_GATE_BIN으로 주입(테스트=fake).
"""
import os, sys, json, hashlib, tempfile, datetime, glob, time, subprocess, re

REPO = os.environ.get("CODEX_GATE_REPO", "/Users/<USER>/Documents/Dacon_236694_AI_Agent")
REPO_REAL = os.path.realpath(REPO)
# COORD_ROOT: 격리 테스트/운영 분리. 미설정 시 production.
COORD = os.path.realpath(os.environ.get("COORD_ROOT", os.path.join(REPO, "open/coordination")))
# 스키마(계약)는 항상 production 고정(테스트도 실제 계약으로 검증). env로 override 가능.
SCHEMA_DIR = os.environ.get("CODEX_GATE_SCHEMA_DIR", os.path.join(REPO, "open/coordination/schemas"))

CODEX = os.environ.get("CODEX_GATE_BIN", "/Applications/ChatGPT.app/Contents/Resources/codex")
MODEL = os.environ.get("CODEX_GATE_MODEL", "gpt-5.5")
REASONING = os.environ.get("CODEX_GATE_REASONING", "xhigh")
POLICY_VERSION = os.environ.get("CODEX_GATE_POLICY", "2026-07-14")
SCHEMA_VERSION = "3"

_TO_OVERRIDE = os.environ.get("CODEX_GATE_TIMEOUT")
TIMEOUTS = {"pre_submit": 1200}
DEFAULT_TIMEOUT = 600
POLL_SEC = float(os.environ.get("CODEX_GATE_POLL", "2"))

STAGES = ["plan_review", "pre_launch", "anomaly_review", "post_gate",
          "post_training", "pre_submit", "post_lb"]
SUBDIRS = ["requests", "results", "results_raw", "prepared", "archive", "locks",
           "state", "logs", "prompts", "status", "snapshots", "infra", "ledger", "cache"]

# 파일수정 감지 대상(러너가 쓰는 coordination 하위는 제외).
MANIFEST_GLOBS = ["CLAUDE.md", "AGENTS.md", "open/docs/**/*.md",
                  "open/scripts/*.py", "scratchpad/*.py"]

SHA_RE = re.compile(r"^[0-9a-f]{64}$")

EXIT = {"PASS": 0, "WAITING_CODEX_APPROVAL": 5, "BLOCK": 10,
        "NEEDS_HUMAN": 20, "INFRA_ERROR": 30, "PREPARED": 5, "GATE_USAGE": 40}

STD_INSTRUCTIONS = (
    "너는 이 대회 프로젝트의 외부 감사자(Codex)다. 사용자가 이 1회 호출을 명시 승인해 실행됐다. 지켜라:\n"
    "1) 읽는 순서: AGENTS.md → open/docs/HANDOFF.md → 관련 실험로그 절 → 대회규칙 → 이 stage 체크리스트.\n"
    "2) Claude(요청자)의 결론을 신뢰하지 말고 실제 파일·evidence_paths를 직접 대조하라.\n"
    "3) 서버 채점은 결정론이다('노이즈' 표현 금지).\n"
    "4) holdout은 제출 자격 필터일 뿐 우열 판단에 쓰지 마라.\n"
    "5) direct_control·intended_change가 단일변수·무교란인지 확인. 동시변경은 confounders에.\n"
    "6) read-only로 확인 불가한 주장은 unverified_claims에. 추정과 사실을 구분하라.\n"
    "7) 어떤 프로젝트 파일도 수정하지 마라. 유일한 출력은 audit_result(raw) JSON이다.\n"
    "8) request_id를 정확히 반향하라.\n"
    "9) safe_next_action은 권고일 뿐 실행 권한이 아니다 — 행동 승인은 사용자만 한다.\n"
    "verdict: 위험단계 진행을 막을 결함=BLOCK, 사용자 결정 필요=NEEDS_HUMAN, 안전=PASS.\n"
)
STAGE_PROMPTS = {
    "plan_review": "실험 계획 감사: 목적·직접 대조군·단일 변경변수·교란·정보가치. 죽은 축 재시도·홀드아웃 우열판단 점검.",
    "pre_launch": "발사 직전: 코드/data SHA frozen 일치·effective_config·split/누수·set_seed 위치·ETA/자원.",
    "anomaly_review": "이상 감사: 크래시·속도저하·환경차이·우회책 부작용이 학습 드로우/계약을 바꿨는지.",
    "post_gate": "홀드아웃/밴드: hidx∩va 정렬·softmax 계약·full-refit 재추론 누수·밴드 자격(자격필터일 뿐).",
    "post_training": "완료: refit 완주·pack↔staging↔zip SHA 체인·manifest·모델 계약(params/head/max_len/fp16).",
    "pre_submit": "제출 직전: verify_zip 전항 + 5행 E2E 근거·기준 diff·비복사본·시간마진. 미통과=BLOCK.",
    "post_lb": "제출 후: 직접 대조 delta·교란 범위·챔피언 판정. '노이즈/단독효과/순수 A/B'가 증거보다 강하면 BLOCK.",
}


# ---------- 기본 유틸 ----------

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(dt=None):
    return (dt or now_utc()).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def d(*p):
    return os.path.join(COORD, *p)


def ensure_dirs():
    for s in SUBDIRS:
        os.makedirs(d(s), exist_ok=True)


def atomic_write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), delete=False,
                                     suffix=".tmp", encoding="utf-8") as tf:
        tf.write(text)
        tmp = tf.name
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2))


def load_schema(name):
    with open(os.path.join(SCHEMA_DIR, name)) as f:
        return json.load(f)


def schema_errors(obj, schema_name):
    """jsonschema 전체 검증 → 위반 메시지 리스트(최대 6). jsonschema 미설치면 예외."""
    import jsonschema
    schema = load_schema(schema_name)
    v = jsonschema.Draft202012Validator(schema)
    out = []
    for e in sorted(v.iter_errors(obj), key=str)[:6]:
        loc = "/".join(map(str, e.path)) or "(root)"
        out.append(f"{loc}: {e.message[:160]}")
    return out


# ---------- 경로/증거 검증 ----------

def resolve_in_repo(p):
    """repo 내부 regular file인지 realpath로 확인. 절대경로 escape·..·symlink escape 차단.
    반환 (real_abs_path, None) 또는 (None, 사유)."""
    full = p if os.path.isabs(p) else os.path.join(REPO, p)
    real = os.path.realpath(full)
    if real != REPO_REAL and not real.startswith(REPO_REAL + os.sep):
        return None, f"repo 밖 경로(escape): {p} → {real}"
    if not os.path.isfile(real):
        return None, f"regular file 아님/미존재: {p}"
    return real, None


def measure_evidence(req):
    """code_sha256/data_sha256 키 + artifact_paths + evidence_paths(+effective_config_path)의
    실측 SHA를 dedup해 {relpath: sha} 반환. 경로검증·SHA형식·SHA대조 실패 시 (None, error_kind, detail)."""
    paths = set()
    for k in ("code_sha256", "data_sha256"):
        paths.update((req.get(k) or {}).keys())
    for k in ("artifact_paths", "evidence_paths"):
        paths.update(req.get(k) or [])
    ecp = req.get("effective_config_path")
    if ecp:
        paths.add(ecp)

    measured = {}
    for p in sorted(paths):
        real, err = resolve_in_repo(p)
        if err:
            return None, "path_escape", err
        measured[os.path.relpath(real, REPO)] = sha256(real)

    # 선언 SHA는 반드시 64-hex이며 실측과 일치해야 함.
    for k in ("code_sha256", "data_sha256"):
        for path, want in (req.get(k) or {}).items():
            if not SHA_RE.match(want or ""):
                return None, "sha_format_invalid", f"{path}: 선언값이 64-hex SHA 아님({want!r}) — 설명은 sha_notes로"
            real, err = resolve_in_repo(path)
            if err:
                return None, "path_escape", err
            got = sha256(real)
            if got != want:
                return None, "evidence_sha_mismatch", f"{path}: req={want[:12]}… 현재={got[:12]}…"
    return measured, None, None


def compute_audit_key(stage, verify_items, policy_version, measured):
    payload = {
        "stage": stage,
        "policy_version": policy_version,
        "verify_items": sorted(verify_items),
        "evidence": sorted([{"path": p, "sha256": s} for p, s in measured.items()],
                           key=lambda e: e["path"]),
    }
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def derive_lineage(req, stage):
    lid = req.get("lineage_id")
    if lid:
        return lid
    # 증거 SHA는 제외(fix→재감사가 같은 lineage 공유하도록).
    return "lin-" + hashlib.sha256(
        canonical({"stage": stage, "verify_items": sorted(req.get("verify_items", []))}).encode()
    ).hexdigest()[:20]


# ---------- 매니페스트(파일수정 감지) ----------

def manifest():
    m = {}
    for g in MANIFEST_GLOBS:
        for p in glob.glob(os.path.join(REPO, g), recursive=True):
            rp = os.path.realpath(p)
            if os.path.isfile(rp) and not rp.startswith(COORD + os.sep):
                m[os.path.relpath(rp, REPO)] = sha256(rp)
    return m


# ---------- 레코드 빌더 ----------

def write_ledger(event, request_id, audit_key, started_at, ended_at, cache_hit,
                 exit_code, tokens, attempt, error_kind=None):
    line = {
        "schema_version": SCHEMA_VERSION, "event": event, "request_id": request_id,
        "audit_key": audit_key, "model": MODEL if event == "codex_call" else None,
        "reasoning": REASONING if event == "codex_call" else None,
        "started_at": started_at, "ended_at": ended_at, "cache_hit": cache_hit,
        "exit_code": exit_code,
        "tokens": tokens or {"input": None, "output": None, "total": None},
        "attempt": attempt, "error_kind": error_kind,
    }
    path = d("ledger", "usage.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def infra_record(req, rid, stage, error_kind, detail, codex_called, audit_key=None,
                 attempt=None, started_at=None, ended_at=None, modified=None, added=None):
    rec = {
        "schema_version": SCHEMA_VERSION, "request_id": rid, "stage": stage,
        "gate_state": "INFRA_ERROR", "action_state": "WAITING_USER",
        "error_kind": error_kind, "error_detail": detail,
        "codex_called": bool(codex_called), "attempt": attempt, "audit_key": audit_key,
        "recorded_at": iso(),
        "runner_recorded": {"model": MODEL if codex_called else None,
                            "reasoning": REASONING if codex_called else None,
                            "started_at": started_at, "ended_at": ended_at},
        "safe_next_action": "위험단계(launch/kill/포장/제출) 진행 금지. 재감사는 새 request_id + 새 사용자 승인 필요.",
    }
    if modified:
        rec["modified_files"] = modified
    if added:
        rec["added_files"] = added
    write_json(d("infra", f"{rid}-{error_kind}.json"), rec)
    return rec


def finding_fingerprint(f):
    fid = f.get("id")
    if fid:
        return str(fid)
    return "fp-" + hashlib.sha1((f.get("summary", "").strip().lower()).encode()).hexdigest()[:12]


def apply_escalation(lineage_id, gate_state, blocking):
    """lineage_id + finding fingerprint의 **연속 미해결 BLOCK**만 누적. 2회 → NEEDS_HUMAN 승격.
    무관 실험(다른 lineage)은 절대 섞이지 않는다(#9). PASS면 해당 lineage 카운터 리셋."""
    st_path = d("state", "escalation.json")
    hist = json.load(open(st_path)) if os.path.exists(st_path) else {}
    escalated = False
    if gate_state == "PASS":
        hist.pop(lineage_id, None)  # 우려 해소 → 리셋
    elif gate_state == "BLOCK":
        lin = hist.setdefault(lineage_id, {})
        for f in blocking:
            fp = finding_fingerprint(f)
            lin[fp] = lin.get(fp, 0) + 1
            if lin[fp] >= 2:
                escalated = True
    write_json(st_path, hist)
    return escalated


def normalize_result(req, rid, stage, audit_key, lineage_id, measured, raw,
                     verdict_source, attempt, started_at, completed_at, raw_ref, cache_hit,
                     gate_state, escalated):
    """★escalation은 여기서 계산하지 않는다(캐시 재사용 시 중복 카운트 방지). 호출자가
    gate_state·escalated를 결정해 전달한다 — 실제 Codex BLOCK 경로에서만 apply_escalation 실행."""
    blocking = raw.get("blocking_findings", [])
    for f in blocking:
        f.setdefault("fingerprint", finding_fingerprint(f))
    res = {
        "schema_version": SCHEMA_VERSION, "request_id": rid, "audit_key": audit_key,
        "lineage_id": lineage_id, "stage": stage, "policy_version": POLICY_VERSION,
        "objective": req.get("objective", raw.get("objective", "")),
        "gate_state": gate_state, "action_state": "WAITING_USER",
        "downstream_action": req.get("downstream_action", "other"),
        "verdict_source": verdict_source, "auditor_model": raw.get("auditor_model", MODEL),
        "cache_hit": cache_hit, "attempt": attempt,
        "started_at": started_at, "completed_at": completed_at,
        "runner_recorded": {"request_id": rid, "model": MODEL, "reasoning": REASONING,
                            "started_at": started_at, "completed_at": completed_at,
                            "attempt": attempt, "cache_hit": cache_hit,
                            "raw_request_id": raw.get("request_id")},
        "raw_result_ref": raw_ref,
        "blocking_findings": blocking,
        "nonblocking_findings": raw.get("nonblocking_findings", []),
        "verified_claims": raw.get("verified_claims", []),
        "unverified_claims": raw.get("unverified_claims", []),
        "confounders": raw.get("confounders", []),
        "required_fixes": raw.get("required_fixes", []),
        "safe_next_action": raw.get("safe_next_action", ""),
        "escalated": escalated,
        "evidence_references": raw.get("evidence_references", []),
        "artifact_sha": measured,
    }
    return res


# ---------- cache ----------

def cache_index_path():
    return d("cache", "index.json")


def cache_lookup(audit_key):
    p = cache_index_path()
    if not os.path.exists(p):
        return None
    idx = json.load(open(p))
    ent = idx.get(audit_key)
    if not ent:
        return None
    rp = d("results", ent["request_id"] + ".json")
    if not os.path.exists(rp):
        return None
    try:
        res = json.load(open(rp))
    except Exception:
        return None
    # 유효 결과만 재사용(인프라 오류는 캐시 안 함 — 애초에 저장 안 됨).
    if res.get("gate_state") in ("PASS", "BLOCK", "NEEDS_HUMAN"):
        return res
    return None


def cache_store(audit_key, rid):
    p = cache_index_path()
    idx = json.load(open(p)) if os.path.exists(p) else {}
    idx[audit_key] = {"request_id": rid, "stored_at": iso()}
    write_json(p, idx)


# ---------- PREPARE ----------

def prepare(req, force_reaudit=False):
    """요청 검증 → 증거 SHA 실측 → audit_key → (cache hit면 재사용) → 프롬프트/prepared 번들 작성.
    ★Codex 미호출. cache hit가 아니면 usage ledger 미기록. 반환 dict(gate_state, exit_code, record)."""
    ensure_dirs()
    stage = req.get("stage")
    rid = req.get("request_id") or f"{stage or 'unknown'}-{now_utc().strftime('%Y%m%dT%H%M%SZ')}-{os.urandom(3).hex()}"
    req.setdefault("request_id", rid)
    req.setdefault("schema_version", SCHEMA_VERSION)
    req.setdefault("policy_version", POLICY_VERSION)
    req.setdefault("created_at", iso())

    # 1) jsonschema 전체 검증(추가필드·타입·SHA pattern 포함) — Codex/실행 전.
    try:
        errs = schema_errors(req, "audit_request.schema.json")
    except ImportError:
        return _fail(req, rid, stage or "unknown", "internal_error", "jsonschema 미설치")
    if errs:
        return _fail(req, rid, stage or "unknown", "request_schema_invalid", "; ".join(errs))

    # 2) 경로/증거 SHA 실측·대조.
    measured, ekind, detail = measure_evidence(req)
    if ekind:
        return _fail(req, rid, stage, ekind, detail)

    audit_key = compute_audit_key(stage, req["verify_items"], req["policy_version"], measured)
    lineage_id = derive_lineage(req, stage)

    # 원자적 요청 게시.
    write_json(d("requests", rid + ".json"), req)

    # 3) cache: 동일 audit_key 유효 결과 재사용(force_reaudit면 건너뜀).
    if not force_reaudit:
        cached = cache_lookup(audit_key)
        if cached:
            started = iso()
            res = normalize_result(
                req, rid, stage, audit_key, lineage_id, measured,
                raw={"verdict": cached["gate_state"], "auditor_model": cached.get("auditor_model", "cache"),
                     "request_id": cached.get("request_id"),
                     "blocking_findings": cached.get("blocking_findings", []),
                     "nonblocking_findings": cached.get("nonblocking_findings", []),
                     "verified_claims": cached.get("verified_claims", []),
                     "unverified_claims": cached.get("unverified_claims", []),
                     "confounders": cached.get("confounders", []),
                     "required_fixes": cached.get("required_fixes", []),
                     "safe_next_action": cached.get("safe_next_action", ""),
                     "evidence_references": cached.get("evidence_references", [])},
                verdict_source="cache", attempt=0, started_at=started, completed_at=iso(),
                raw_ref=d("results", cached["request_id"] + ".json"), cache_hit=True,
                gate_state=cached["gate_state"], escalated=cached.get("escalated", False))
            res["objective"] = req.get("objective", res["objective"])
            _self_validate_and_persist(res, rid)
            write_ledger("cache_hit", rid, audit_key, started, iso(), True, None,
                         {"input": None, "output": None, "total": None}, 0)
            _archive(req, res)
            return {"gate_state": res["gate_state"], "exit_code": EXIT[res["gate_state"]],
                    "record": res, "cache_hit": True, "audit_key": audit_key}

    # 4) 프롬프트 + prepared 번들(승인 대기). Codex 미호출.
    prompt = (STD_INSTRUCTIONS + "\n[STAGE=" + stage + "] " + STAGE_PROMPTS.get(stage, "") +
              "\n\n[AUDIT_REQUEST]\n" + json.dumps(req, ensure_ascii=False, indent=2) +
              "\n\n출력: audit_result(raw) 스키마를 정확히 따르는 JSON만. request_id=" + rid)
    atomic_write(d("prompts", rid + ".txt"), prompt)
    bundle = {
        "schema_version": SCHEMA_VERSION, "request_id": rid, "stage": stage,
        "policy_version": req["policy_version"], "audit_key": audit_key, "lineage_id": lineage_id,
        "gate_state": "WAITING_CODEX_APPROVAL", "action_state": "WAITING_USER",
        "downstream_action": req.get("downstream_action", "other"),
        "objective": req.get("objective", ""), "measured_evidence": measured,
        "prompt_path": d("prompts", rid + ".txt"), "prepared_at": iso(),
        "codex_call_requires_user_approval": True,
        "note": "PREPARE_ONLY 완료. Codex 호출은 사용자 명시 승인 후 execute 단계에서만 1회. AUDIT_PASS는 행동 승인과 별개.",
    }
    write_json(d("prepared", rid + ".json"), bundle)
    return {"gate_state": "WAITING_CODEX_APPROVAL", "exit_code": EXIT["WAITING_CODEX_APPROVAL"],
            "record": bundle, "cache_hit": False, "audit_key": audit_key}


def _fail(req, rid, stage, ekind, detail):
    rec = infra_record(req, rid, stage, ekind, detail, codex_called=False)
    return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"],
            "record": rec, "error_kind": ekind, "detail": detail}


def _self_validate_and_persist(res, rid):
    errs = schema_errors(res, "audit_result.schema.json")
    if errs:
        raise RuntimeError(f"러너 정규화 결과가 result 스키마 위반(internal): {errs}")
    write_json(d("results", rid + ".json"), res)


def _archive(req, res):
    write_json(d("archive", res["request_id"] + ".json"), {"request": req, "result": res})


# ---------- Codex 실행 ----------

def _parse_tokens(jsonl_path):
    """codex --json 이벤트에서 token usage 파싱. 없으면 모두 null(추정 금지)."""
    tok = {"input": None, "output": None, "total": None}
    if not os.path.exists(jsonl_path):
        return tok
    try:
        for line in open(jsonl_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            usage = None
            if isinstance(ev, dict):
                usage = ev.get("usage") or (ev.get("msg", {}) if isinstance(ev.get("msg"), dict) else {}).get("usage")
            if isinstance(usage, dict):
                tok["input"] = usage.get("input_tokens", usage.get("prompt_tokens", tok["input"]))
                tok["output"] = usage.get("output_tokens", usage.get("completion_tokens", tok["output"]))
                tok["total"] = usage.get("total_tokens", tok["total"])
    except Exception:
        pass
    return tok


def _run_codex_once(prompt, raw_path, jsonl_path, err_path, timeout):
    """Codex subprocess를 **정확히 1회** 기동. 자동 재시도 없음. 반환 (raw_dict|None, note, exit_code)."""
    cmd = [CODEX, "exec", "--json", "--model", MODEL, "-c", f'model_reasoning_effort="{REASONING}"',
           "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check", "-C", REPO,
           "--output-schema", os.path.join(SCHEMA_DIR, "audit_result.codex.schema.json"),
           "--output-last-message", raw_path, prompt]
    if os.path.exists(raw_path):
        os.remove(raw_path)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    jf = open(jsonl_path, "w", buffering=1, encoding="utf-8")
    ef = open(err_path, "w", buffering=1, encoding="utf-8")
    start = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=jf, stderr=ef, cwd=REPO)
    timed_out = False
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if time.time() - start > timeout:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            timed_out = True
            rc = proc.poll()
            break
        time.sleep(POLL_SEC)
    jf.close(); ef.close()
    if timed_out:
        return None, f"codex timeout {timeout}s", rc
    if rc != 0:
        stderr_tail = ""
        try:
            stderr_tail = open(err_path, encoding="utf-8").read()[-400:]
        except Exception:
            pass
        return None, f"codex exit {rc}: {stderr_tail}", rc
    if not os.path.exists(raw_path):
        return None, "결과 파일 미생성", rc
    try:
        return json.load(open(raw_path)), None, rc
    except json.JSONDecodeError as e:
        return None, f"결과 JSON 파싱 실패: {e}", rc


def execute(bundle, approval, force_reaudit=False):
    """★명시 승인 정보가 있을 때만 Codex를 **정확히 1회** 실행. 승인은 1호출 유효.
    approval = {"approved": True, "request_id": ..., "audit_key": ...} (사용자 명시).
    반환 dict(gate_state, exit_code, record)."""
    ensure_dirs()
    rid = bundle["request_id"]
    stage = bundle["stage"]
    audit_key = bundle["audit_key"]
    lineage_id = bundle["lineage_id"]
    req_path = d("requests", rid + ".json")
    req = json.load(open(req_path)) if os.path.exists(req_path) else {"request_id": rid, "stage": stage}

    # 1) 승인 검증 — 없거나 불일치면 Codex 미호출·ledger 미기록.
    if not (approval and approval.get("approved") is True):
        rec = infra_record(req, rid, stage, "approval_missing",
                           "Codex 호출 사용자 승인 없음 — execute 거부(기본 PREPARE_ONLY)", codex_called=False,
                           audit_key=audit_key)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}
    if approval.get("request_id") != rid or approval.get("audit_key") != audit_key:
        rec = infra_record(req, rid, stage, "approval_mismatch",
                           f"승인 대상 불일치(approval rid/key ≠ bundle) — Codex 미호출", codex_called=False,
                           audit_key=audit_key)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}

    # 2) 이미 실행됨(승인 소진) — force_reaudit라도 승인 자체는 위에서 요구됨.
    executed_marker = d("locks", rid + ".executed")
    if os.path.exists(executed_marker) and not force_reaudit:
        rec = infra_record(req, rid, stage, "already_executed",
                           "이 승인은 이미 1회 소진됨 — 재감사는 새 request_id + 새 승인 필요", codex_called=False,
                           audit_key=audit_key)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}

    # 3) 증거 재검증(prepare 이후 파일이 바뀌지 않았는지) — 바뀌면 Codex 미호출.
    measured, ekind, detail = measure_evidence(req)
    if ekind:
        rec = infra_record(req, rid, stage, ekind, detail, codex_called=False, audit_key=audit_key)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}
    if compute_audit_key(stage, req["verify_items"], req.get("policy_version", POLICY_VERSION), measured) != audit_key:
        rec = infra_record(req, rid, stage, "evidence_sha_mismatch",
                           "prepare 이후 증거가 변경됨(audit_key 불일치) — 재prepare 필요", codex_called=False,
                           audit_key=audit_key)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}

    # 4) Codex 1회 실행(재시도 없음). attempt 로그 분리.
    prompt = open(bundle["prompt_path"], encoding="utf-8").read()
    attempt = 1
    log_dir = d("logs", rid)
    os.makedirs(log_dir, exist_ok=True)
    jsonl_path = os.path.join(log_dir, "attempt-01.jsonl")
    err_path = os.path.join(log_dir, "attempt-01.stderr.log")
    raw_path = d("results_raw", rid + ".json")

    before = manifest()
    started_at = iso()
    raw, note, rc = _run_codex_once(prompt, raw_path, jsonl_path, err_path,
                                    int(req.get("timeout_sec") or _TO_OVERRIDE or TIMEOUTS.get(stage, DEFAULT_TIMEOUT)))
    ended_at = iso()
    after = manifest()

    # 승인 소진 마킹(성공/실패 무관 — 1호출 유효).
    atomic_write(executed_marker, json.dumps({"executed_at": ended_at, "attempt": attempt}))

    tokens = _parse_tokens(jsonl_path)
    write_ledger("codex_call", rid, audit_key, started_at, ended_at, False, rc, tokens, attempt,
                 error_kind=None if raw is not None else "run_failure")

    # 5) 파일수정 감지 → INFRA_ERROR(codex_modified_files). raw와 무관하게 우선.
    modified = [p for p in before if before.get(p) != after.get(p)]
    added = [p for p in after if p not in before]
    if modified or added:
        rec = infra_record(req, rid, stage, "codex_modified_files",
                           f"read-only인데 파일 수정 감지 — 강제 fail-closed. modified={modified} added={added}",
                           codex_called=True, audit_key=audit_key, attempt=attempt,
                           started_at=started_at, ended_at=ended_at, modified=modified, added=added)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}

    # 6) 실행 실패(exit≠0/timeout/미생성/파싱실패) → INFRA_ERROR(재시도 없음).
    if raw is None:
        kind = "codex_timeout" if "timeout" in (note or "") else \
               ("codex_http_error" if "400" in (note or "") else "codex_exit_nonzero")
        if "미생성" in (note or "") or "파싱" in (note or ""):
            kind = "result_missing"
        rec = infra_record(req, rid, stage, kind, note or "codex 실패", codex_called=True,
                           audit_key=audit_key, attempt=attempt, started_at=started_at, ended_at=ended_at)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}

    # 7) raw 스키마 전체 검증 — 부적합이면 PASS로 승격 절대 금지 → INFRA_ERROR.
    rerrs = schema_errors(raw, "audit_result.codex.schema.json")
    if rerrs:
        rec = infra_record(req, rid, stage, "result_schema_invalid",
                           f"raw 결과 스키마 위반 — 정규화 거부: {rerrs}", codex_called=True,
                           audit_key=audit_key, attempt=attempt, started_at=started_at, ended_at=ended_at)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}
    if raw.get("request_id") != rid:
        rec = infra_record(req, rid, stage, "result_rid_mismatch",
                           f"raw request_id 불일치(stale): {raw.get('request_id')} != {rid}", codex_called=True,
                           audit_key=audit_key, attempt=attempt, started_at=started_at, ended_at=ended_at)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}

    # 8) escalation(실제 Codex BLOCK 경로에서만) → 정규화 + 자기검증 + 영속화 + cache 저장.
    verdict = raw["verdict"]
    blocking = raw.get("blocking_findings", [])
    escalated = apply_escalation(lineage_id, verdict, blocking)
    gate_state = "NEEDS_HUMAN" if (verdict == "BLOCK" and escalated) else verdict
    res = normalize_result(req, rid, stage, audit_key, lineage_id, measured, raw,
                           verdict_source="codex", attempt=attempt, started_at=started_at,
                           completed_at=ended_at, raw_ref=raw_path, cache_hit=False,
                           gate_state=gate_state, escalated=escalated)
    try:
        _self_validate_and_persist(res, rid)
    except RuntimeError as e:
        rec = infra_record(req, rid, stage, "internal_error", str(e), codex_called=True,
                           audit_key=audit_key, attempt=attempt, started_at=started_at, ended_at=ended_at)
        return {"gate_state": "INFRA_ERROR", "exit_code": EXIT["INFRA_ERROR"], "record": rec}
    cache_store(audit_key, rid)
    _archive(req, res)
    bundle["gate_state"] = res["gate_state"]
    write_json(d("prepared", rid + ".json"), bundle)
    return {"gate_state": res["gate_state"], "exit_code": EXIT[res["gate_state"]], "record": res}
