#!/usr/bin/env python3
"""H100 sk의 GroupKFold(5) fold0 va가 치타 hidx(=밴드체크 기준행)와 일치하는지 검증.
스크립트 line243과 동일: next(GroupKFold(5).split(texts[:N_REAL], y[:N_REAL], groups[:N_REAL]))."""
import sys, numpy as np, sklearn
sys.path.insert(0, ".")
import feat
from sklearn.model_selection import GroupKFold

real = feat.load_jsonl("data/train.jsonl")
N = len(real)
ids = [s["id"] for s in real]
groups = np.array([feat.session_of(i) for i in ids])
y = np.zeros(N, dtype=int)  # split은 y값 무시
X = np.zeros(N)
tr, va = next(GroupKFold(n_splits=5).split(X, y, groups))
va = np.array(va)
hidx = np.load("hidx.npy")
print(f"sklearn={sklearn.__version__} N_REAL={N} va_len={len(va)} hidx_len={len(hidx)}")
print(f"order-exact-equal: {np.array_equal(va, hidx)}")
print(f"set-equal: {np.array_equal(np.sort(va), np.sort(hidx))}")
print(f"va[:5]={va[:5]} hidx[:5]={hidx[:5]}")
