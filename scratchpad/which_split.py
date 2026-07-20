"""hp_*.npy 멤버가 hidx split인지 GKF-fold0 split인지 실측 (F1로 결판)."""
import json, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
sys.path.insert(0, "open/scripts")
import feat

train = feat.load_jsonl("open/data/train.jsonl")
ids = [s["id"] for s in train]
N = len(ids)
groups = np.array([feat.session_of(i) for i in ids])
labdf = pd.read_csv("open/data/train_labels.csv")
lab = dict(zip(labdf["id"], labdf["action"]))
ACT = feat.ACTIONS
lab2id = {a:i for i,a in enumerate(ACT)}
y = np.array([lab2id[lab[i]] for i in ids])

hidx = np.load("scratchpad/hidx.npy").astype(np.int64).ravel()
_, gkf_va = next(GroupKFold(5).split(ids, y, groups))  # colab 순서 그대로(sort 안 함)

hp79 = np.load("scratchpad/hp_kf768_79.npy")
print("hp_kf768_79 shape:", hp79.shape, "dtype", hp79.dtype)
pred79 = hp79.argmax(1)  # argmax는 softmax 불변

def f1_with(idx_map):
    if len(idx_map) != len(pred79): return None
    return f1_score(y[idx_map], pred79, average="macro")

print(f"k79 F1 (hidx 정렬):      {f1_with(hidx):.4f}" if len(hidx)==len(pred79) else f"len mismatch hidx {len(hidx)} vs pred {len(pred79)}")
print(f"k79 F1 (GKF-fold0 정렬): {f1_with(gkf_va):.4f}")
print(f"  (recon 단독 홀드아웃 k79 = 0.7617 과 대조)")

# hidx vs hybrid/holdout_idx3 동일성
hyb = np.load("open/artifacts/hybrid/holdout_idx3.npy").astype(np.int64).ravel()
print(f"hidx == hybrid/holdout_idx3.npy ? {np.array_equal(hidx, hyb)}  (hidx {len(hidx)} / hyb {len(hyb)})")
# hidx가 정렬돼있나 (colab va는 비정렬)
print(f"hidx 정렬돼있나? {np.array_equal(hidx, np.sort(hidx))}")
print(f"gkf_va 정렬돼있나? {np.array_equal(gkf_va, np.sort(gkf_va))}")
