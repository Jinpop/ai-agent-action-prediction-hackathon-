#!/usr/bin/env python3
"""감사 요청/결과/인프라 스키마 검증기 (2026-07-14 재설계).

★legacy 판정은 **mtime cutoff가 아니라** schema_version + 명시적 backfilled + legacy_manifest로만 처리(#6).
- schema_version=="3" 파일  → 해당 v3 스키마로 **엄격 검증**. 위반 = 즉시 exit 1(신규 위반 은닉 불가).
- backfilled==true 파일      → 소급 기록. 최신 스키마로 검증하되 위반은 backfill 경고(exit 0)로만.
- schema_version 미표기 파일  → legacy_manifest.json에 있으면 legacy(v2 관용, exit 0). 목록 밖이면
                              "신규인데 schema_version 누락" 위반(exit 1) — mtime으로 숨을 수 없다.

검사: jsonschema 전체 + request_id↔파일명 + 결과↔짝 request + (v3) INFRA_ERROR는 infra 스키마로.
COORD_ROOT env로 격리 검증 가능(운영 오염 없이 테스트).

사용: [COORD_ROOT=...] open/.venv/bin/python scratchpad/validate_audits.py
종료코드: 0=발효 이후 파일 전부 통과, 1=위반 존재
"""
import json, os, sys, glob

ROOT = os.path.realpath(os.environ.get(
    "COORD_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "open", "coordination")))
SCHEMA_DIR = os.environ.get("CODEX_GATE_SCHEMA_DIR", os.path.join(ROOT, "schemas"))
# 스키마는 항상 production 계약을 쓸 수 있게: COORD_ROOT가 격리라도 schemas가 없으면 production로 폴백.
if not os.path.isdir(SCHEMA_DIR):
    SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "open", "coordination", "schemas")

try:
    import jsonschema
    HAVE_JS = True
except ImportError:
    HAVE_JS = False


def load(p):
    with open(p) as f:
        return json.load(f)


def schema_viol(doc, schema):
    if not HAVE_JS:
        return []
    return [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message[:140]}"
            for e in sorted(jsonschema.Draft202012Validator(schema).iter_errors(doc), key=str)[:5]]


def main():
    req_schema = load(os.path.join(SCHEMA_DIR, "audit_request.schema.json"))
    res_schema = load(os.path.join(SCHEMA_DIR, "audit_result.schema.json"))
    infra_schema = load(os.path.join(SCHEMA_DIR, "infra_error.schema.json"))

    legacy_files = set()
    lm = os.path.join(ROOT, "legacy_manifest.json")
    if os.path.exists(lm):
        legacy_files = set(load(lm).get("files", {}).keys())  # "requests/xxx.json" 형태

    problems, legacy, backfilled = [], [], []
    reqs = {os.path.basename(p): p for p in glob.glob(os.path.join(ROOT, "requests", "*.json"))}
    ress = {os.path.basename(p): p for p in glob.glob(os.path.join(ROOT, "results", "*.json"))}
    infras = {os.path.basename(p): p for p in glob.glob(os.path.join(ROOT, "infra", "*.json"))}

    def classify(rel, doc):
        """legacy 판정은 mtime이 아니라 legacy_manifest + schema_version + backfilled로만(#6).
        우선순위: (1) manifest 등재 = pre-v3 frozen → wholesale 관용 (2) backfilled=true = 명시 소급기록 → soft
        (3) schema_version=='3' → v3 엄격 (4) 그 외(미표기 & 목록 밖 & 비backfilled) → 위반."""
        if rel in legacy_files:
            return legacy
        if doc.get("backfilled") is True:
            return backfilled
        if doc.get("schema_version") == "3":
            return problems
        return problems  # schema_version 미표기 + 목록 밖 + 비backfilled = 신규 누락 → 위반

    def check(kind, name, path, schema):
        rel = ("requests/" if kind == "request" else "results/") + name
        try:
            doc = load(path)
        except Exception as e:
            problems.append(f"[{kind}] {name}: JSON 파싱 실패 — {e}")
            return
        sink = classify(rel, doc)
        # legacy(미표기 & manifest 등재)는 wholesale 관용 — v3 스키마로 재검증하지 않는다(소급수정 금지 정신).
        if sink is legacy:
            legacy.append(f"[{kind}] {name}: legacy(schema_version 미표기, manifest 등재) — v3 재검증 생략")
            return
        if doc.get("schema_version") is None and not doc.get("backfilled"):
            sink.append(f"[{kind}] {name}: schema_version 미표기 + legacy_manifest 밖 → 신규 기록은 schema_version='3' 필수")
        for v in schema_viol(doc, schema):
            sink.append(f"[{kind}] {name}: 스키마 위반 — {v}")
        rid = doc.get("request_id", "")
        if rid and rid != name[:-5]:
            sink.append(f"[{kind}] {name}: request_id '{rid}' ≠ 파일명")

    for name, path in sorted(reqs.items()):
        check("request", name, path, req_schema)
    for name, path in sorted(ress.items()):
        check("result", name, path, res_schema)

    # infra (v3 전용)
    for name, path in sorted(infras.items()):
        try:
            doc = load(path)
        except Exception as e:
            problems.append(f"[infra] {name}: JSON 파싱 실패 — {e}")
            continue
        for v in schema_viol(doc, infra_schema):
            problems.append(f"[infra] {name}: infra 스키마 위반 — {v}")

    # 결과 ↔ 짝 request (파일명 동일 기준. legacy 결과는 짝 없어도 관용)
    for name in sorted(ress):
        if name in reqs:
            continue
        rel = "results/" + name
        if rel in legacy_files or (name[:-5] and False):
            legacy.append(f"[pair] results/{name}: 짝 request 없음(legacy 관용)")
        else:
            try:
                doc = load(ress[name])
            except Exception:
                doc = {}
            if doc.get("schema_version") == "3":
                problems.append(f"[pair] results/{name}: 짝 request 없음(v3는 requests/ 게시 필수)")
            else:
                legacy.append(f"[pair] results/{name}: 짝 request 없음(pre-v3)")

    if not HAVE_JS:
        problems.append("[env] jsonschema 미설치 — 전체 스키마 검증 생략됨 (open/.venv에 설치 필요)")

    print(f"ROOT={ROOT}")
    print(f"요청 {len(reqs)} / 결과 {len(ress)} / infra {len(infras)} 검사  (legacy_manifest {len(legacy_files)}건)")
    if legacy:
        print(f"\n[legacy — schema_version 미표기 & manifest 등재, 소급수정 안 함] {len(legacy)}건:")
        for p in legacy:
            print(" ~", p)
    if backfilled:
        print(f"\n[backfilled — 명시 소급기록] {len(backfilled)}건:")
        for p in backfilled:
            print(" ·", p)
    if problems:
        print(f"\n★위반(v3 엄격 또는 미표기 신규 — 수정 필수) {len(problems)}건:")
        for p in problems:
            print(" -", p)
        return 1
    print("\nv3/신규 파일 전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
