"""read-only: colab GroupKFold-fold0 va vs scratchpad/hidx.npy — 동일 split인지 실측."""
import json, sys
import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "open/scripts")
import feat

# colab과 동일 순서로 real train 로드
train = feat.load_jsonl("open/data/train.jsonl")
ids = [s["id"] for s in train]
N_REAL = len(ids)
groups = np.array([feat.session_of(i) for i in ids])
y = np.zeros(N_REAL, dtype=int)  # GroupKFold는 y 미사용

tr, va = next(GroupKFold(n_splits=5).split(ids, y, groups))
va = np.sort(va)

hidx = np.load("scratchpad/hidx.npy")
hidx = np.sort(hidx.astype(np.int64).ravel())

print(f"N_REAL={N_REAL}")
print(f"GroupKFold fold0 va: {len(va)} rows, {len(set(groups[va]))} sessions")
print(f"hidx.npy:            {len(hidx)} rows, {len(set(groups[hidx]))} sessions" if hidx.max()<N_REAL else f"hidx.npy: {len(hidx)} rows (max idx {hidx.max()} >= N_REAL!)")
same_rows = np.array_equal(va, hidx)
print(f"row-index identical? {same_rows}")
if not same_rows and hidx.max()<N_REAL:
    va_sess = set(groups[va]); hidx_sess = set(groups[hidx])
    inter = va_sess & hidx_sess
    print(f"session overlap: {len(inter)} / va {len(va_sess)} / hidx {len(hidx_sess)}")
    print(f"row overlap: {len(np.intersect1d(va, hidx))}")
