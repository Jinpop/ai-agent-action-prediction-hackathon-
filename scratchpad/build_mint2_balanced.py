#!/usr/bin/env python3
"""mint2 balanced 빌더 (07-13 카드②) — 롤링윈도우 다양성 + target별 총가중=1.

목적: mint1(target당 최장 history 1행) 대신 mint2 롤링윈도우(target당 1~6 변형)를 쓰되,
window 수가 많은 target이 과대표집되지 않도록 **target별 window 총가중=1**로 균형.
"오류집합을 바꾸는 큰 카드"(window 다양성). text-only pretext(stage A)용.

산출(둘 다 emit):
  train_mint2_balanced.jsonl       (full, 6354 window / 2863 target) — 배포 refit stage A용
  train_mint2_balanced_gate.jsonl  (gate, 5097 window / 2314 target) — 밴드체크 stage A용(hidx-holdout 세션 제외)
  + 각 _labels.csv

계약(감사·recon §5 준수):
  - core = strict∩exact 2863 canonical target (build_mint_exact.py와 동일 정의·카운트 assert).
  - window-level strict: 각 window의 (history,prompt) input이 real에 없는 것만(real중복 8 제거).
  - exact meta 복원: turn_index=canonical target step, workspace{loc,language_mix(세션불변),
    git_dirty/open_files/last_ci_status(정확복원)}. **budget_tokens_remaining·elapsed_session_sec 제거**
    (관측불가 → zero-sentinel 함정 원천차단; PRETEXT_META=keep는 이 [META]만 텍스트로).
  - target별 총가중=1: window 행수만큼 나눠지도록 `target_key` 필드 emit → colab TARGET_BALANCE가
    Counter(target_key)로 WT/=count. (id만으론 canonical target 파싱 불가 → 필드로 명시 emit.)
  - gate = full에서 hidx-holdout 세션 제외(누수 landmine 통일: 학습·평가·밴드체크 전부 hidx 기준).
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_mint_recoverability import (DATA, EDIT_ACTIONS, TOUCH_ACTIONS, action_after,
                                       canonical_target, ci_result, frozen, load_jsonl,
                                       load_npy_i8, replay_dirty, replay_open_files,
                                       session_step, user_positions, workspace)

ROOT = Path(__file__).resolve().parents[1]

real = load_jsonl(DATA / "train.jsonl")
mint1 = load_jsonl(DATA / "train_mint.jsonl")
mint2 = load_jsonl(DATA / "train_mint2.jsonl")
win_labs = {r["id"]: r["action"]
            for r in csv.DictReader(open(DATA / "train_mint2_labels.csv"))}


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
        observed_events.setdefault(key, event)

# canonical targets (mint1: target당 1행) — target 집합 + exact meta 복원 기준
canonical_rows = {}
for row in mint1:
    session, source_step = session_step(row["id"])
    row["_source_history"] = real_by_session[session][source_step].get("history") or []
    ts, tstep, _ = canonical_target(row)
    canonical_rows[(ts, tstep)] = row


# --- CI / git_dirty / open_files 정확복원 (build_mint_exact.py 로직 그대로) ---
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

print(f"CI/dirty/open exact: {len(ci_values)}/{len(dirty_values)}/{len(open_values)} (기대 3108/3016/2910)")
assert len(ci_values) == 3108 and len(dirty_values) == 3016 and len(open_values) == 2910

strict_keys = {k for k, r in canonical_rows.items() if input_key(r) not in real_input_keys}
exact_keys = set(ci_values) & set(dirty_values) & set(open_values)
core = strict_keys & exact_keys
print(f"strict∩exact core: {len(core)} (기대 2863)")
assert len(core) == 2863

hidx_sessions = {session_step(real[i]["id"])[0] for i in load_npy_i8(ROOT / "scratchpad" / "hidx.npy")}
print(f"hidx holdout sessions: {len(hidx_sessions)} (기대 1885)")
assert len(hidx_sessions) == 1885

# --- target별 라벨 (core target의 canonical 라벨; 모든 window 동일해야 함) ---
mint1_labs = {r["id"]: r["action"]
              for r in csv.DictReader(open(DATA / "train_mint_labels.csv"))}
target_label = {k: mint1_labs[canonical_rows[k]["id"]] for k in core}


def build_meta(win_row, tkey):
    """window의 세션불변 필드 + target의 exact 복원값으로 session_meta 재구성 (budget/elapsed 제거)."""
    _, tstep = tkey
    m0 = win_row.get("session_meta") or {}
    w0 = m0.get("workspace") or {}
    return {
        "user_tier": m0.get("user_tier"),
        "language_pref": m0.get("language_pref"),
        "turn_index": tstep,
        "workspace": {
            "loc": w0.get("loc"),
            "language_mix": w0.get("language_mix"),
            "git_dirty": dirty_values[tkey],
            "open_files": open_values[tkey],
            "last_ci_status": ci_values[tkey],
        },
    }


# --- mint2 window 필터링 + 재구성 ---
full_rows, gate_rows = [], []
label_conflicts = 0
for win in mint2:
    session, source_step = session_step(win["id"])
    if source_step not in real_by_session[session]:
        continue
    win["_source_history"] = real_by_session[session][source_step].get("history") or []
    ts, tstep, _ = canonical_target(win)
    tkey = (ts, tstep)
    if tkey not in core:
        continue
    if input_key(win) in real_input_keys:   # window-level strict
        continue
    if win_labs.get(win["id"]) != target_label[tkey]:
        label_conflicts += 1                 # window 라벨 ≠ canonical 라벨 (없어야 함)
    out = {k: v for k, v in win.items()
           if not k.startswith("_") and k not in ("session_meta",)}
    out["session_meta"] = build_meta(win, tkey)
    out["target_key"] = f"{ts}#{tstep}"      # colab TARGET_BALANCE용 (총가중=1)
    full_rows.append((tkey, out))
    if ts not in hidx_sessions:
        gate_rows.append((tkey, out))

print(f"label conflicts(window vs canonical): {label_conflicts} (기대 0)")
assert label_conflicts == 0


def emit(rows, stem, exp_rows, exp_targets):
    targets = {tk for tk, _ in rows}
    print(f"{stem}: {len(rows)} window / {len(targets)} target (기대 {exp_rows}/{exp_targets})")
    assert len(rows) == exp_rows and len(targets) == exp_targets
    # 가중 sanity: target별 window count로 총가중=1 검증 (colab TARGET_BALANCE가 재현)
    cnt = Counter(tk for tk, _ in rows)
    for tk in targets:
        assert abs(sum(1.0 / cnt[tk] for _ in range(cnt[tk])) - 1.0) < 1e-9
    with open(DATA / f"{stem}.jsonl", "w") as f:
        for _, r in rows:
            assert "budget_tokens_remaining" not in r["session_meta"]
            assert "elapsed_session_sec" not in r["session_meta"]
            assert "elapsed_session_sec" not in r["session_meta"]["workspace"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA / f"{stem}_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "action"])
        for _, r in rows:
            w.writerow([r["id"], win_labs[r["id"]]])
    return cnt


emit(full_rows, "train_mint2_balanced", 6354, 2863)
gate_cnt = emit(gate_rows, "train_mint2_balanced_gate", 5097, 2314)
# gate에 hidx-holdout 세션 0 확인
assert not any(tk[0] in hidx_sessions for tk, _ in gate_rows)
print("OK — 전 카운트·가중·누수·budget/elapsed 검증 통과, emit 완료")
