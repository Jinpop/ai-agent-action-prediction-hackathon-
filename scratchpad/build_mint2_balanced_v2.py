#!/usr/bin/env python3
"""mint2 balanced **v2** 빌더 (07-14 외부감사 반영).

v1(build_mint2_balanced.py, 2,863 target/6,354 window) 대비 변경:
  ① core = strict 전체(exact-meta 복원 조건 제거) — 외부감사: "메타 3종 미복원 이유로
     299 target·475 unique window를 버렸으나 그 필드는 [META]에 쓰이지도 않았음".
     기대 3,162 target / 6,829 unique window.
  ② 내부 중복 제거: 선택된 window 전체에서 (history,prompt) input 전역 유일화.
     tiebreak = target-커버리지 우선(window 수 적은 target이 input 우선 점유 — 유일-window
     target 소멸 방지), 동률이면 id 사전순 최소(결정론).
  ③ 메타 = 전부 omit 방침(사용자 옵션 중 택1): 학습은 PRETEXT_META=omit로 [META] 라인
     자체가 없음. session_meta는 window 자체의 세션불변·정확 필드만 남김
     (user_tier/language_pref/workspace.loc/language_mix — 복원·조작·zero-sentinel 없음).
     git_dirty/open_files/ci/turn/budget/elapsed 전부 미기재(feat는 .get() 기본값, PRETEXT가 119d 0화).
  ④ target_key emit 유지(TARGET_BALANCE) — 학습은 TB_V2=1(배치독립 가중손실·target가중 클래스가중)과 병용.

산출:
  train_mint2_balanced_v2.jsonl / _labels.csv          (full — 배포 stage A용)
  train_mint2_balanced_v2_gate.jsonl / _labels.csv     (gate — hidx-holdout 세션 제외)
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_mint_recoverability import (DATA, canonical_target, frozen, load_jsonl,
                                       load_npy_i8, session_step)

ROOT = Path(__file__).resolve().parents[1]

real = load_jsonl(DATA / "train.jsonl")
mint1 = load_jsonl(DATA / "train_mint.jsonl")
mint2 = load_jsonl(DATA / "train_mint2.jsonl")
win_labs = {r["id"]: r["action"]
            for r in csv.DictReader(open(DATA / "train_mint2_labels.csv"))}
mint1_labs = {r["id"]: r["action"]
              for r in csv.DictReader(open(DATA / "train_mint_labels.csv"))}


def input_key(row):
    return frozen(row.get("history") or []), row.get("current_prompt", "")


real_input_keys = {input_key(row) for row in real}

real_by_session = defaultdict(dict)
for row in real:
    session, step = session_step(row["id"])
    real_by_session[session][step] = row

# canonical targets (mint1: target당 1행)
canonical_rows = {}
for row in mint1:
    session, source_step = session_step(row["id"])
    row["_source_history"] = real_by_session[session][source_step].get("history") or []
    ts, tstep, _ = canonical_target(row)
    canonical_rows[(ts, tstep)] = row

# ① strict 전체 (exact-meta 조건 없음)
strict_keys = {k for k, r in canonical_rows.items() if input_key(r) not in real_input_keys}
print(f"strict targets: {len(strict_keys)} (기대 3162)")
assert len(strict_keys) == 3162, f"strict target 수 불일치: {len(strict_keys)}"

hidx_sessions = {session_step(real[i]["id"])[0] for i in load_npy_i8(ROOT / "scratchpad" / "hidx.npy")}
assert len(hidx_sessions) == 1885

target_label = {k: mint1_labs[canonical_rows[k]["id"]] for k in strict_keys}

STABLE_META_KEYS = ("user_tier", "language_pref")
STABLE_WS_KEYS = ("loc", "language_mix")


def stable_meta(win_row):
    """window 자체의 세션불변·정확 필드만 (복원·기본값 채움 없음 — 없는 키는 그대로 없음)."""
    m0 = win_row.get("session_meta") or {}
    w0 = m0.get("workspace") or {}
    meta = {k: m0[k] for k in STABLE_META_KEYS if k in m0}
    ws = {k: w0[k] for k in STABLE_WS_KEYS if k in w0}
    if ws:
        meta["workspace"] = ws
    return meta


# --- window 수집: target strict + window strict ---
picked = []       # (input_key, id, tkey, out_row)
label_conflicts = 0
for win in mint2:
    session, source_step = session_step(win["id"])
    if source_step not in real_by_session[session]:
        continue
    win["_source_history"] = real_by_session[session][source_step].get("history") or []
    ts, tstep, _ = canonical_target(win)
    tkey = (ts, tstep)
    if tkey not in strict_keys:
        continue
    if input_key(win) in real_input_keys:   # window-level strict (real 중복 제거)
        continue
    if win_labs.get(win["id"]) != target_label[tkey]:
        label_conflicts += 1
        continue                            # 라벨 충돌 window는 배제(카운트 보고)
    out = {k: v for k, v in win.items()
           if not k.startswith("_") and k not in ("session_meta",)}
    out["session_meta"] = stable_meta(win)
    out["target_key"] = f"{ts}#{tstep}"
    picked.append((input_key(win), win["id"], tkey, out))

print(f"label conflicts(배제): {label_conflicts}")

# ② 내부 중복 제거 — 동일 input 전역 유일화. tiebreak = **target 커버리지 우선**
# (window 수가 적은 target이 input을 우선 점유 — 유일-window target 소멸 방지, 07-14 진단:
#  040088#1의 유일 window가 id-min 규칙에선 타 target 중복에 밀려 target 3,161로 감소),
# 동률이면 id 사전순(결정론).
_tw_cnt = Counter(tkey for _, _, tkey, _ in picked)
picked.sort(key=lambda t: (_tw_cnt[t[2]], t[1]))
seen_inputs = {}          # ikey -> (kept_id, kept_label)
full_rows = []
dup_dropped = 0
conflicts = []            # 동일 input·교차라벨 상세(감사 기록용)
for ikey, rid, tkey, out in picked:
    if ikey in seen_inputs:
        dup_dropped += 1
        kept_id, kept_label = seen_inputs[ikey]
        if kept_label != target_label[tkey]:
            conflicts.append({"kept_id": kept_id, "kept_label": kept_label,
                              "dropped_id": rid, "dropped_label": target_label[tkey]})
        continue
    seen_inputs[ikey] = (rid, target_label[tkey])
    full_rows.append((tkey, out))
full_rows.sort(key=lambda t: t[1]["id"])

print(f"내부중복 제거: {dup_dropped}행, 그중 교차라벨(모호 input) {len(conflicts)}건 — target-커버리지 우선(동률 id-min) 유지·상세 기록")
for c in conflicts:
    print(f"  ⚠️ 동일 input 교차라벨: keep {c['kept_id']}({c['kept_label']}) / drop {c['dropped_id']}({c['dropped_label']})")
if conflicts:
    with open(ROOT / "scratchpad" / "mint2_v2_label_conflicts.json", "w") as f:
        json.dump(conflicts, f, ensure_ascii=False, indent=2)
    # dedup 후 학습셋 내 모순은 없음(한쪽만 유지). 모호 input의 라벨 선택은 id-min 결정론 —
    # 총 6,829행 중 소수라 영향 미미하나 plan_review 감사에 명시 전달할 것.

gate_rows = [(tk, r) for tk, r in full_rows if tk[0] not in hidx_sessions]


def emit(rows, stem):
    targets = {tk for tk, _ in rows}
    print(f"{stem}: {len(rows)} window / {len(targets)} target")
    with open(DATA / f"{stem}.jsonl", "w") as f:
        for _, r in rows:
            sm = r["session_meta"]
            assert "budget_tokens_remaining" not in sm and "elapsed_session_sec" not in sm
            assert "git_dirty" not in sm.get("workspace", {}) and "last_ci_status" not in sm.get("workspace", {})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA / f"{stem}_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "action"])
        for _, r in rows:
            w.writerow([r["id"], win_labs[r["id"]]])
    return len(rows), len(targets)


n_full, t_full = emit(full_rows, "train_mint2_balanced_v2")
n_gate, t_gate = emit(gate_rows, "train_mint2_balanced_v2_gate")
assert not any(tk[0] in hidx_sessions for tk, _ in gate_rows)

# 기대치 검증(외부감사 수치): full = 6,829 window / 3,162 target
print(f"기대 대조 — full: {n_full}/{t_full} (외부감사 기대 6829/3162), gate: {n_gate}/{t_gate}")
assert t_full == 3162, f"full target {t_full} ≠ 3162"
if n_full != 6829:
    print(f"⚠️ full window {n_full} ≠ 기대 6829 — 원인 규명 전 발사 금지 (dedup 규칙 차이 가능)")
    sys.exit(2)
print("OK — v2 emit 완료 (strict 3,162 target · unique 6,829 window · 안정필드-only meta · 누수 gate 분리)")
