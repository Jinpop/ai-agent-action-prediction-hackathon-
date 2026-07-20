#!/usr/bin/env python3
"""Fair holdout eval of kf1024_74 vs kf768_74 in the neural core (audit §11 method).
Per-member softmax -> mean -> argmax. Reports row Macro-F1 + session-equal Macro-F1.
Sanity: reproduce audit's s48+s51+kf768_74 = 0.7588 / 0.7654."""
import sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "open/scripts")
import feat
from sklearn.metrics import f1_score

def load_labels(p):
    d = {}
    for row in csv.DictReader(open(p)):
        d[row["id"]] = row["action"]
    return d

ACT = feat.ACTIONS
a2id = {a: i for i, a in enumerate(ACT)}
real = feat.load_jsonl("open/data/train.jsonl")
rlab = load_labels("open/data/train_labels.csv")
ids = [s["id"] for s in real]
y = np.array([a2id[rlab[i]] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])

# hidx.npy is canonical (local sklearn GroupKFold differs from training env — known trap)
hidx = np.load("scratchpad/hidx.npy")
hidx_kf1024 = np.load("scratchpad/hidx_kf1024_74.npy")
assert np.array_equal(hidx, hidx_kf1024), "kf1024 hidx mismatch"
y_va = y[hidx]
g_va = groups[hidx]

# session-equal weights: each session contributes equally
import collections
cnt = collections.Counter(g_va)
w = np.array([1.0 / cnt[g] for g in g_va])

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def as_probs(a, name):
    s = a.sum(axis=1)
    if np.all(a >= 0) and np.allclose(s, 1.0, atol=1e-3):
        print(f"  {name}: already probs")
        return a
    print(f"  {name}: raw logits (row-sum ex: {s[:2]}) -> softmax")
    return softmax(a.astype(np.float64))

def load(name, path):
    a = np.load(path)
    assert a.shape == (len(hidx), 14), f"{name} shape {a.shape}"
    assert np.isfinite(a).all(), f"{name} non-finite"
    return as_probs(a, name)

s48 = load("s48", "open/artifacts/hybrid/holdout_probs_s48.npy")
s51 = load("s51", "open/artifacts/hybrid/holdout_probs_s51.npy")
kf768 = load("kf768_74", "scratchpad/hp_kf768_74.npy")
kf1024 = load("kf1024_74", "scratchpad/hp_kf1024_74.npy")

def rep(name, probs):
    pred = probs.argmax(axis=1)
    row = f1_score(y_va, pred, average="macro")
    se = f1_score(y_va, pred, average="macro", sample_weight=w)
    print(f"{name:34s} row={row:.4f}  sess-eq={se:.4f}")
    return row

print()
rep("kf768_74 solo", kf768)
rep("kf1024_74 solo", kf1024)
base = rep("s48+s51+kf768_74 (v45 core)", (s48 + s51 + kf768) / 3)
new = rep("s48+s51+kf1024_74", (s48 + s51 + kf1024) / 3)
print(f"\ncore delta (kf1024 swap): {new - base:+.4f}  (band gate: >= -0.0058)")
