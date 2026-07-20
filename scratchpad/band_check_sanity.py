#!/usr/bin/env python3
"""H100 런 SANITY gate (누수 없는 정식 방식): canonical hidx ∩ H100 va 교집합에서
양쪽 진짜 out-of-fold holdout logits만 정렬해 계산.
- s48/k79/k78/k81: hidx 기반 hp (홀드아웃 모델 out-of-fold, 누수 없음)
- H100 신멤버: va 기반 holdout_probs3 (홀드아웃 모델 out-of-fold, 누수 없음)
- 교집합 2825행/379세션 → sanity(정식 14000 gate보다 표본 작음, 노이즈 큼).
usage: band_check_sanity.py [<run_holdout_probs3.npy> <run_holdout_idx3.npy> <label>]
인자 없으면 베이스라인만. 챔피언 v58(s48+k79+k81) 기준 delta >= -0.0058(sanity).
"""
import sys, csv, numpy as np
sys.path.insert(0, "open/scripts")
import feat
from sklearn.metrics import f1_score

hidx = np.load("scratchpad/hidx.npy")
va = np.load("scratchpad/h100_va.npy")
inter = np.intersect1d(hidx, va)
hmap = {int(r): i for i, r in enumerate(hidx)}
vmap = {int(r): i for i, r in enumerate(va)}
hi = np.array([hmap[int(r)] for r in inter])
vi = np.array([vmap[int(r)] for r in inter])

real = feat.load_jsonl("open/data/train.jsonl")
rlab = {r["id"]: r["action"] for r in csv.DictReader(open("open/data/train_labels.csv"))}
ACT = feat.ACTIONS; a2id = {a: i for i, a in enumerate(ACT)}
y = np.array([a2id[rlab[real[int(r)]["id"]]] for r in inter])
g = np.array([feat.session_of(real[int(r)]["id"]) for r in inter])
import collections
cnt = collections.Counter(g); w = np.array([1.0 / cnt[x] for x in g])

def sm(z):
    z = z.astype(np.float64); z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)
def loadhidx(p): return sm(np.load(p)[hi])  # hidx 기반 → 교집합 위치
s48 = loadhidx("open/artifacts/hybrid/holdout_probs_s48.npy")
k79 = loadhidx("scratchpad/hp_kf768_79.npy")
k78 = loadhidx("scratchpad/hp_kf768_78.npy")
k81 = loadhidx("scratchpad/hp_kf768_81.npy")
def se(core): return f1_score(y, core.argmax(1), average="macro", sample_weight=w)

b58 = se((s48 + k79 + k81) / 3)   # 신챔피언 코어
b49 = se((s48 + k79 + k78) / 3)   # 구 코어(참고)
print(f"[SANITY inter={len(inter)}행/{len(set(g))}세션]")
print(f"baseline v58-core(s48+k79+k81) = {b58:.4f} (gate={b58-0.0058:.4f})")
print(f"baseline v49-core(s48+k79+k78) = {b49:.4f} (참고)")

if len(sys.argv) >= 3:
    probs_path, idx_path = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else probs_path
    ridx = np.load(idx_path)
    if not np.array_equal(np.sort(ridx), np.sort(va)):
        print(f"[ABORT] {label}: holdout_idx3 집합이 h100_va와 불일치 — 밴드 무효"); sys.exit(2)
    # 런 va가 순서 다르면 런 자체 매핑으로 교집합 정렬
    rmap = {int(r): i for i, r in enumerate(ridx)}
    ri = np.array([rmap[int(r)] for r in inter])
    raw = np.load(probs_path)
    assert raw.shape[0] == len(ridx) and raw.shape[1] == 14 and np.isfinite(raw).all()
    new = sm(raw[ri])
    cand = se((s48 + k79 + new) / 3); solo = se(new)
    d58 = cand - b58
    print(f"\n=== {label} (SANITY) ===")
    print(f"solo(new) sess-eq = {solo:.4f}")
    print(f"candidate s48+k79+new = {cand:.4f}")
    print(f"delta vs v58-core = {d58:+.4f}   (sanity gate -0.0058: {'PASS' if d58>=-0.0058 else 'FAIL'})")
    print("※ 자격필터(sanity)일 뿐 우열 아님. 최종 판정은 서버 LB.")
