"""고전 3모델(HGB/LogReg/NB)의 5-fold OOF 확률 생성 — 강교사 재료.

- fold마다 벡터라이저/SVD/모델 전부 fold-train만으로 재학습 (누수 없음)
- held fold 예측을 배포 블렌드 비율(0.45/0.40/0.15)로 합성
- 출력: artifacts/oof/oof_classic_probs.npy (N,14 ACTIONS순) + oof_classic_ids.json
- 참고: fold 분할이 seed OOF(H100 sklearn)와 달라도 무방 — 각자 out-of-fold이기만 하면 교사 조립에 문제 없음
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feat
from train import make_vectorizers, make_hgb, make_logreg, make_nb, transform_text

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "oof")
W3 = (0.45, 0.40, 0.15)
SVD_COMPONENTS = 200

t0 = time.time()
samples = feat.load_jsonl(os.path.join(DATA, "train.jsonl"))
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
label_map = dict(zip(labels["id"], labels["action"]))
ids = [s["id"] for s in samples]
y = np.array([label_map[i] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])
texts = [feat.build_text(s) for s in samples]
meta_df = feat.build_meta_frame(samples)
Xmeta_all = meta_df.values.astype(np.float32)
ACTIONS = list(feat.ACTIONS)
print(f"[{time.time()-t0:.0f}s] 데이터 준비 {len(samples)}행", flush=True)

oof = np.full((len(samples), len(ACTIONS)), np.nan, np.float32)
for k, (tr, va) in enumerate(GroupKFold(5).split(texts, y, groups)):
    tk = time.time()
    vecs = make_vectorizers()
    tr_texts = [texts[i] for i in tr]
    for v in vecs:
        v.fit(tr_texts)
    Xs_tr = transform_text(vecs, tr_texts)
    Xs_va = transform_text(vecs, [texts[i] for i in va])
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42).fit(Xs_tr)
    scaler = MaxAbsScaler().fit(Xmeta_all[tr])
    Xhgb_tr = np.hstack([svd.transform(Xs_tr).astype(np.float32), Xmeta_all[tr]])
    Xhgb_va = np.hstack([svd.transform(Xs_va).astype(np.float32), Xmeta_all[va]])
    Xlr_tr = sp_hstack([Xs_tr, csr_matrix(scaler.transform(Xmeta_all[tr]))]).tocsr()
    Xlr_va = sp_hstack([Xs_va, csr_matrix(scaler.transform(Xmeta_all[va]))]).tocsr()

    sw = compute_sample_weight("balanced", y[tr])
    hgb = make_hgb().fit(Xhgb_tr, y[tr], sample_weight=sw)
    lr = make_logreg().fit(Xlr_tr, y[tr])
    nb = make_nb().fit(Xs_tr, y[tr])
    classes = list(hgb.classes_)
    p = (W3[0] * hgb.predict_proba(Xhgb_va) + W3[1] * lr.predict_proba(Xlr_va)
         + W3[2] * nb.predict_proba(Xs_va))
    remap = [classes.index(a) for a in ACTIONS]
    oof[va] = p[:, remap]
    f1 = f1_score(y[va], np.array(ACTIONS)[oof[va].argmax(1)], average="macro")
    print(f"[{time.time()-t0:.0f}s] fold {k+1}/5 완료 ({time.time()-tk:.0f}s) heldF1={f1:.4f}", flush=True)

assert not np.isnan(oof).any()
os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, "oof_classic_probs.npy"), oof)
json.dump(ids, open(os.path.join(OUT, "oof_classic_ids.json"), "w"))
f1_all = f1_score(y, np.array(ACTIONS)[oof.argmax(1)], average="macro")
print(f"[{time.time()-t0:.0f}s] 저장 완료. classic OOF 전체 macroF1={f1_all:.4f}", flush=True)
