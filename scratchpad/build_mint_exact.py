#!/usr/bin/env python3
"""exact-core mint 데이터 빌더 (Codex audit_mint_recoverability.py 로직 재현 + emit).

산출:
  open/data/train_mint_exact.jsonl        (2,863행 기대) — session_meta를 '정확 복원 가능 필드만'으로 재구성
  open/data/train_mint_exact_gate.jsonl   (2,314행 기대) — 위에서 홀드아웃 세션 제외 (gate용)
  + 각 _labels.csv

session_meta(실험군 [META]용): user_tier/language_pref(세션불변), turn_index=canonical target step(증명됨),
workspace{loc, language_mix(세션불변), last_ci_status/git_dirty/open_files(정확복원)}.
budget_tokens_remaining/elapsed_session_sec는 관측 불가 → 제외(감사 결론).
Codex 공표 카운트(3108/3016/2910/2863/2314)와 전부 일치해야 emit."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_mint_recoverability import (DATA, EDIT_ACTIONS, TOUCH_ACTIONS, action_after,
                                       canonical_target, ci_result, frozen, load_jsonl,
                                       load_npy_i8, replay_dirty, replay_open_files,
                                       session_step, user_positions, workspace)

ROOT = Path(__file__).resolve().parents[1]

real = load_jsonl(DATA / "train.jsonl")
mint1 = load_jsonl(DATA / "train_mint.jsonl")
labs = {r["id"]: r["action"] for r in csv.DictReader(open(DATA / "train_mint_labels.csv"))}


def input_key(row):
    return frozen(row.get("history") or []), row.get("current_prompt", "")


real_input_keys = {input_key(row) for row in real}

real_by_session = defaultdict(dict)
for row in real:
    session, step = session_step(row["id"])
    real_by_session[session][step] = row

observed_events = {}
for row in real:
    session, source_step = session_step(row["id"])
    history = row.get("history") or []
    positions = user_positions(history)
    base_step = source_step - len(positions)
    for ui, pos in enumerate(positions):
        event = action_after(history, pos)
        if event is None:
            continue
        key = (session, base_step + ui)
        if key not in observed_events:
            observed_events[key] = event

canonical_rows = {}
for row in mint1:
    session, source_step = session_step(row["id"])
    source = real_by_session[session][source_step]
    row["_source_history"] = source.get("history") or []
    target_session, target_step, _ = canonical_target(row)
    canonical_rows[(target_session, target_step)] = row

# --- CI 정확복원 (감사 스크립트 로직 그대로) ---
ci_values = {}
for key in sorted(canonical_rows):
    session, target = key
    observed = real_by_session[session]
    derivations = []
    for anchor in sorted((s for s in observed if s < target), reverse=True):
        value = workspace(observed[anchor]).get("last_ci_status")
        complete = True
        for step in range(anchor, target):
            event = observed_events.get((session, step))
            if event is None:
                complete = False
                break
            update = ci_result(event)
            if update is not None:
                value = update
        if complete and value is not None:
            derivations.append(("forward", value))
            break
    for anchor in sorted(s for s in observed if s > target):
        complete = True
        changed = False
        for step in range(target, anchor):
            event = observed_events.get((session, step))
            if event is None:
                complete = False
                break
            if event.get("name") == "run_tests":
                changed = True
        if complete and not changed:
            value = workspace(observed[anchor]).get("last_ci_status")
            if value is not None:
                derivations.append(("backward_no_test", value))
            break
    values = {v for _, v in derivations}
    if len(values) == 1:
        ci_values[key] = values.pop()


# --- git_dirty / open_files 정확복원 (감사 스크립트 recover_dynamic 그대로) ---
def recover_dynamic(field):
    recovered = {}
    for key in sorted(canonical_rows):
        session, target = key
        observed = real_by_session[session]
        derivations = []
        for anchor in sorted((s for s in observed if s < target), reverse=True):
            if field == "git_dirty":
                value = bool(workspace(observed[anchor]).get(field))
            else:
                value = list(workspace(observed[anchor]).get(field) or [])
            complete = True
            for step in range(anchor, target):
                event = observed_events.get((session, step))
                if event is None:
                    complete = False
                    break
                value = replay_dirty(value, event) if field == "git_dirty" else replay_open_files(value, event)
            if complete:
                derivations.append(("forward", value))
                break
        for anchor in sorted(s for s in observed if s > target):
            events = []
            complete = True
            for step in range(target, anchor):
                event = observed_events.get((session, step))
                if event is None:
                    complete = False
                    break
                events.append(event)
            if not complete:
                continue
            if field == "git_dirty":
                later = bool(workspace(observed[anchor]).get(field))
                if not later:
                    derivations.append(("backward_false", False))
                elif not any(e.get("name") in EDIT_ACTIONS for e in events):
                    derivations.append(("backward_no_edit", True))
            elif not any(e.get("name") in TOUCH_ACTIONS for e in events):
                derivations.append(("backward_no_touch", list(workspace(observed[anchor]).get(field) or [])))
            break
        values = {frozen(v): v for _, v in derivations}
        if len(values) == 1:
            recovered[key] = next(iter(values.values()))
    return recovered


dirty_values = recover_dynamic("git_dirty")
open_values = recover_dynamic("open_files")

# --- 카운트 검증 (Codex 공표치와 전부 일치해야 진행) ---
print(f"CI exact: {len(ci_values)} (기대 3108)")
print(f"dirty exact: {len(dirty_values)} (기대 3016)")
print(f"open exact: {len(open_values)} (기대 2910)")
assert len(ci_values) == 3108 and len(dirty_values) == 3016 and len(open_values) == 2910

strict_keys = {k for k, r in canonical_rows.items() if input_key(r) not in real_input_keys}
exact_keys = set(ci_values) & set(dirty_values) & set(open_values)
core = sorted(strict_keys & exact_keys)
print(f"strict∩exact: {len(core)} (기대 2863)")
assert len(core) == 2863

holdout_sessions = {session_step(real[i]["id"])[0] for i in load_npy_i8(ROOT / "scratchpad" / "hidx.npy")}
gate = [k for k in core if k[0] not in holdout_sessions]
print(f"gate(홀드아웃 제외): {len(gate)} (기대 2314)")
assert len(gate) == 2314


def emit(keys, stem):
    rows_out = []
    for key in keys:
        session, target = key
        src = canonical_rows[key]
        m0 = src.get("session_meta") or {}
        w0 = m0.get("workspace") or {}
        row = {k: v for k, v in src.items() if not k.startswith("_") and k != "session_meta"}
        row["session_meta"] = {
            "user_tier": m0.get("user_tier"),
            "language_pref": m0.get("language_pref"),
            "turn_index": target,               # 증명: 실 데이터에서 turn≡step
            "workspace": {
                "loc": w0.get("loc"),
                "language_mix": w0.get("language_mix"),
                "git_dirty": dirty_values[key],
                "open_files": open_values[key],
                "last_ci_status": ci_values[key],
            },  # budget/elapsed 관측불가 → 미포함
        }
        rows_out.append(row)
    with open(DATA / f"{stem}.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA / f"{stem}_labels.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["id", "action"])
        for r in rows_out:
            wcsv.writerow([r["id"], labs[r["id"]]])
    print(f"emit {stem}: {len(rows_out)}행")


emit(core, "train_mint_exact")
emit(gate, "train_mint_exact_gate")
print("OK — 전 카운트 일치, emit 완료")
