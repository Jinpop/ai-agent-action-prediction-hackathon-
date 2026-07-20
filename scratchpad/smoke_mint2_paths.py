"""GPU 없이 colab 신규경로 로직 검증: TARGET_BALANCE 가중 + HOLDOUT_IDX split + 누수불변식."""
import json, sys
from collections import Counter
import numpy as np
sys.path.insert(0, "open/scripts")
import feat

# --- 1) TARGET_BALANCE: gate jsonl로 WT 재현 (colab 로직 그대로) ---
rows = [json.loads(l) for l in open("open/data/train_mint2_balanced_gate.jsonl")]
ids = [r["id"] for r in rows]
target_keys = [r.get("target_key") for r in rows]
assert all(t is not None for t in target_keys), "target_key 누락"
WT = np.ones(len(ids), np.float32)
_tcnt = Counter(target_keys)
WT *= np.array([1.0/_tcnt[t] for t in target_keys], np.float32)
WT /= WT.mean()
# 불변식: target별 WT 합이 전부 동일(=총가중 균등). 과표집 제거 검증.
per_t = {}
for w, t in zip(WT, target_keys):
    per_t[t] = per_t.get(t, 0.0) + w
sums = np.array(list(per_t.values()))
print(f"[TARGET_BALANCE] {len(_tcnt)} target, {len(ids)} window")
print(f"  target별 WT합: min {sums.min():.6f} max {sums.max():.6f} std {sums.std():.2e}  (전부 동일해야)")
print(f"  window별 WT: min {WT.min():.4f} max {WT.max():.4f} (window많은 target일수록 개별 작음)")
assert sums.std() < 1e-5, "target별 총가중 불균등!"
# window 6개 target과 1개 target의 개별가중 비 = 1:6 인지
c1 = [t for t,c in _tcnt.items() if c==1][0]; c6 = [t for t,c in _tcnt.items() if c==max(_tcnt.values())][0]
w1 = WT[target_keys.index(c1)]; w6 = WT[target_keys.index(c6)]
print(f"  window1 target 개별가중 {w1:.4f} vs window{max(_tcnt.values())} target 개별가중 {w6:.4f} → 비 {w1/w6:.2f}")

# --- 2) HOLDOUT_IDX split: hidx pin 재현 ---
N_REAL = 70000
va = np.load("scratchpad/hidx.npy").astype(int).ravel()
tr = np.setdiff1d(np.arange(N_REAL), va)
print(f"\n[HOLDOUT_IDX] va={len(va)} tr={len(tr)} overlap={len(np.intersect1d(tr,va))} union={len(set(tr)|set(va))}")
assert len(va)==14000 and len(tr)==56000 and len(np.intersect1d(tr,va))==0 and len(set(tr)|set(va))==N_REAL

# --- 3) ★누수 불변식: gate-mint(stage A) 세션 ∩ hidx-holdout(stage B eval) 세션 = 0 ---
real = feat.load_jsonl("open/data/train.jsonl")
hidx_sessions = {feat.session_of(real[i]["id"]) for i in va}
gate_sessions = {feat.session_of(r["id"]) for r in rows}
leak = gate_sessions & hidx_sessions
print(f"\n[누수불변식] gate-mint 세션 {len(gate_sessions)}, hidx-holdout 세션 {len(hidx_sessions)}, 교집합 {len(leak)} (0이어야)")
assert len(leak)==0, f"누수! gate mint에 hidx-holdout 세션 {len(leak)}개 포함"

# --- 4) budget/elapsed 원천부재 (gate jsonl 전행) ---
bad = 0
for r in rows:
    sm = r.get("session_meta") or {}
    if "budget_tokens_remaining" in sm or "elapsed_session_sec" in sm: bad += 1
    if "elapsed_session_sec" in (sm.get("workspace") or {}): bad += 1
print(f"\n[budget/elapsed 부재] 위반 행 {bad} (0이어야)")
assert bad==0
print("\nOK — TARGET_BALANCE 균등·HOLDOUT_IDX split정합·누수0·budget/elapsed부재 전부 통과")
