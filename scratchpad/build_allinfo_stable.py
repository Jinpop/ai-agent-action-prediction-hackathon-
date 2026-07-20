#!/usr/bin/env python3
"""ALL-INFO Stage A stable-meta pretext 데이터 빌드.
검증된 train_mint2_balanced_v2(6829 window / 3162 target / real-dup 18제외 / target_key)에
turn_index(원천 raw mint2에서 100% 복원가능)만 세션메타에 조인 주입 = 5-field stable.
그 외 필드/dedup/가중은 전부 v2 그대로(1변수 변경). sentinel 대체 없음(없는 키는 생략).
"""
import json, ast, os, shutil, collections
ROOT = "/Users/<USER>/Documents/Dacon_236694_AI_Agent"
RAW = f"{ROOT}/open/data/train_mint2.jsonl"                    # 원천(turn_index 보유)
SRC = f"{ROOT}/open/data/train_mint2_balanced_v2.jsonl"        # 검증된 6829 stable(4필드)
SRC_LAB = f"{ROOT}/open/data/train_mint2_balanced_v2_labels.csv"
OUT = f"{ROOT}/open/data/train_mint_allinfo_stable.jsonl"
OUT_LAB = f"{ROOT}/open/data/train_mint_allinfo_stable_labels.csv"
FROZEN = f"{ROOT}/scratchpad/allinfo_frozen/colab_train_base2.py"

# 1) 원천에서 id -> turn_index 맵
turn_by_id = {}
with open(RAW) as f:
    for line in f:
        r = json.loads(line)
        sm = r.get("session_meta") or {}
        if "turn_index" in sm and sm["turn_index"] is not None:
            turn_by_id[r["id"]] = sm["turn_index"]
print(f"[raw] turn_index 보유 id: {len(turn_by_id)}")

# 2) balanced_v2 각 행에 turn_index 주입 (순서 보존, 그 외 무변경)
rows = []
missing_turn = 0
field_cov = collections.Counter()
seen_ids = set()
with open(SRC) as f:
    for line in f:
        r = json.loads(line)
        rid = r["id"]
        assert rid not in seen_ids, f"중복 id {rid}"
        seen_ids.add(rid)
        sm = r.get("session_meta") or {}
        assert "turn_index" not in sm, "이미 turn_index 존재 — 이중주입"
        if rid in turn_by_id:
            # 세션메타 최상위에 turn_index 삽입(원천 구조와 동일 위치)
            new_sm = {}
            if "user_tier" in sm: new_sm["user_tier"] = sm["user_tier"]
            if "language_pref" in sm: new_sm["language_pref"] = sm["language_pref"]
            new_sm["turn_index"] = turn_by_id[rid]
            if "workspace" in sm: new_sm["workspace"] = sm["workspace"]
            r["session_meta"] = new_sm
        else:
            missing_turn += 1  # 복원 불가 → sentinel 없이 turn 생략(그대로 둠)
        sm2 = r.get("session_meta") or {}
        ws2 = sm2.get("workspace") or {}
        if "user_tier" in sm2: field_cov["user_tier"] += 1
        if "language_pref" in sm2: field_cov["language_pref"] += 1
        if "turn_index" in sm2: field_cov["turn_index"] += 1
        if "loc" in ws2: field_cov["loc"] += 1
        if ws2.get("language_mix"): field_cov["language_mix"] += 1
        rows.append(r)

# 3) 계약 검증
n = len(rows)
tkeys = {r.get("target_key") for r in rows}
print(f"[out] rows={n} (기대 6829)  distinct target_key={len(tkeys)} (기대 3162)")
assert n == 6829, f"window 수 불일치 {n}"
assert len(tkeys) == 3162, f"target 수 불일치 {len(tkeys)}"
assert all(r.get("target_key") for r in rows), "target_key 누락 행 존재"
assert missing_turn == 0, f"turn_index 복원 실패 {missing_turn}행 (manifest 5필드 위반)"
print(f"[cov] 필드 커버리지(6829 중): {dict(field_cov)}")

# 4) 라벨 복사(행 순서 동일 → 라벨 정렬 유지)
shutil.copyfile(SRC_LAB, OUT_LAB)

# 5) 출력
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[write] {OUT} ({n}행), 라벨 {OUT_LAB}")

# 6) frozen build_transcript 로 실제 transcript 샘플 증명 (fail-closed #4)
src = open(FROZEN).read()
tree = ast.parse(src)
ns = {"os": os, "PRETEXT_META": "stable", "META_TRANS_EXT": False}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("_s", "build_transcript"):
        exec(compile(ast.Module([node], []), FROZEN, "exec"), ns)
bt = ns["build_transcript"]
print("\n=== 실제 frozen build_transcript(PRETEXT_META=stable) 샘플 3건의 [META] 라인 ===")
shown = 0
for r in rows:
    t = bt(r)
    meta_lines = [ln for ln in t.split("\n") if ln.startswith("[META]")]
    assert len(meta_lines) == 1, f"[META] 라인 수 이상: {len(meta_lines)}"
    ml = meta_lines[0]
    # sentinel 금지 검증: 없는 키가 0/None/빈값으로 새지 않았는지 (budget/elapsed/ci/dirty/open 미포함)
    for forbidden in ("budget=", "elapsed=", "ci=", "dirty=", "open="):
        assert forbidden not in ml, f"Stage A [META]에 금지필드 {forbidden} 누출: {ml}"
    if shown < 3:
        print(f"  id={r['id']}: {ml}")
        shown += 1
print("[proof] 전 6829행 [META] = stable 5필드만, 금지필드 누출 0 ✓")
