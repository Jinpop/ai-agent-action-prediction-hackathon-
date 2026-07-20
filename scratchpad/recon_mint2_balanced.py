"""read-only 카운트 정찰 — mint2 balanced 빌더 설계용. emit 없음.
audit_mint_recoverability 로직 재사용: canonical target별 exact meta 복원 → mint2 window를
core-2863 필터·window-level strict·hidx-holdout 제외로 세어 빌더 assertion 수치 확보."""
import csv, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, "scratchpad")
from audit_mint_recoverability import (DATA, EDIT_ACTIONS, TOUCH_ACTIONS, action_after,
    canonical_target, ci_result, frozen, load_jsonl, load_npy_i8, replay_dirty,
    replay_open_files, session_step, user_positions, workspace)
ROOT = Path(".").resolve()

real = load_jsonl(DATA / "train.jsonl")
mint1 = load_jsonl(DATA / "train_mint.jsonl")
mint2 = load_jsonl(DATA / "train_mint2.jsonl")

def input_key(row):
    return frozen(row.get("history") or []), row.get("current_prompt", "")
real_input_keys = {input_key(r) for r in real}

real_by_session = defaultdict(dict)
for r in real:
    s, st = session_step(r["id"]); real_by_session[s][st] = r

observed_events = {}
for r in real:
    s, src = session_step(r["id"]); h = r.get("history") or []
    positions = user_positions(h); base = src - len(positions)
    for ui, pos in enumerate(positions):
        ev = action_after(h, pos)
        if ev is None: continue
        k = (s, base + ui)
        observed_events.setdefault(k, ev)

# canonical targets from mint1 (target set + source linkage)
canonical_rows = {}
for r in mint1:
    s, src = session_step(r["id"])
    r["_source_history"] = real_by_session[s][src].get("history") or []
    ts, tstep, _ = canonical_target(r)
    canonical_rows[(ts, tstep)] = r
print(f"canonical targets(mint1): {len(canonical_rows)} (기대 3180)")

# --- exact meta 복원 (build_mint_exact.py 로직 그대로) ---
ci_values = {}
for key in sorted(canonical_rows):
    session, target = key; observed = real_by_session[session]; derivations = []
    for anchor in sorted((s for s in observed if s < target), reverse=True):
        value = workspace(observed[anchor]).get("last_ci_status"); complete = True
        for step in range(anchor, target):
            ev = observed_events.get((session, step))
            if ev is None: complete = False; break
            up = ci_result(ev)
            if up is not None: value = up
        if complete and value is not None: derivations.append(("f", value)); break
    for anchor in sorted(s for s in observed if s > target):
        complete = True; changed = False
        for step in range(target, anchor):
            ev = observed_events.get((session, step))
            if ev is None: complete = False; break
            if ev.get("name") == "run_tests": changed = True
        if complete and not changed:
            value = workspace(observed[anchor]).get("last_ci_status")
            if value is not None: derivations.append(("b", value))
        break
    vals = {v for _, v in derivations}
    if len(vals) == 1: ci_values[key] = vals.pop()

def recover_dynamic(field):
    rec = {}
    for key in sorted(canonical_rows):
        session, target = key; observed = real_by_session[session]; derivations = []
        for anchor in sorted((s for s in observed if s < target), reverse=True):
            value = bool(workspace(observed[anchor]).get(field)) if field=="git_dirty" else list(workspace(observed[anchor]).get(field) or [])
            complete = True
            for step in range(anchor, target):
                ev = observed_events.get((session, step))
                if ev is None: complete = False; break
                value = replay_dirty(value, ev) if field=="git_dirty" else replay_open_files(value, ev)
            if complete: derivations.append(("f", value)); break
        for anchor in sorted(s for s in observed if s > target):
            events = []; complete = True
            for step in range(target, anchor):
                ev = observed_events.get((session, step))
                if ev is None: complete = False; break
                events.append(ev)
            if not complete: continue
            if field=="git_dirty":
                later = bool(workspace(observed[anchor]).get(field))
                if not later: derivations.append(("bf", False))
                elif not any(e.get("name") in EDIT_ACTIONS for e in events): derivations.append(("bne", True))
            elif not any(e.get("name") in TOUCH_ACTIONS for e in events):
                derivations.append(("bnt", list(workspace(observed[anchor]).get(field) or [])))
            break
        vals = {frozen(v): v for _, v in derivations}
        if len(vals) == 1: rec[key] = next(iter(vals.values()))
    return rec

dirty_values = recover_dynamic("git_dirty"); open_values = recover_dynamic("open_files")
print(f"CI/dirty/open exact: {len(ci_values)}/{len(dirty_values)}/{len(open_values)} (기대 3108/3016/2910)")
strict_keys = {k for k,r in canonical_rows.items() if input_key(r) not in real_input_keys}
exact_keys = set(ci_values) & set(dirty_values) & set(open_values)
core = strict_keys & exact_keys
print(f"strict∩exact core: {len(core)} (기대 2863)")

hidx_sessions = {session_step(real[i]["id"])[0] for i in load_npy_i8(ROOT/"scratchpad"/"hidx.npy")}
print(f"hidx holdout sessions: {len(hidx_sessions)}")

# --- mint2 windows: canonical target 매핑 + 필터 카운트 ---
win_per_target = Counter()      # 모든 mint2 window의 canonical target 분포
win_core = 0                    # core 타겟에 속하는 window
win_core_strict = 0            # + window-level strict(input ∉ real)
win_core_strict_gate = 0      # + hidx-holdout 세션 제외
win_dupreal = 0               # window input == real (제거 대상)
targets_covered = set(); targets_covered_gate = set()
bad_canon = 0
for r in mint2:
    s, src = session_step(r["id"])
    if src not in real_by_session[s]: bad_canon += 1; continue
    r["_source_history"] = real_by_session[s][src].get("history") or []
    ts, tstep, _ = canonical_target(r); tkey = (ts, tstep)
    win_per_target[tkey] += 1
    if tkey not in core: continue
    win_core += 1
    if input_key(r) in real_input_keys: win_dupreal += 1; continue
    win_core_strict += 1
    targets_covered.add(tkey)
    if ts not in hidx_sessions:
        win_core_strict_gate += 1; targets_covered_gate.add(tkey)

print("\n=== mint2 window 카운트 ===")
print(f"mint2 총 window: {len(mint2)} (기대 6859), source 못찾음: {bad_canon}")
print(f"core 타겟 window: {win_core}")
print(f"  - window-level strict(input∉real): {win_core_strict}  (real중복 제거 {win_dupreal})")
print(f"  - gate(hidx-holdout 세션 제외): {win_core_strict_gate}")
print(f"커버 타겟수: full={len(targets_covered)}/{len(core)}  gate={len(targets_covered_gate)}")
ws = [win_per_target[t] for t in core]
print(f"core 타겟당 window 분포: min {min(ws)} max {max(ws)} mean {np.mean(ws):.2f} (합 {sum(ws)})")
# core인데 window 0개인 타겟(mint1엔 있으나 mint2엔 없음)
zero = [t for t in core if win_per_target[t]==0]
print(f"core인데 mint2 window 0개: {len(zero)}")
