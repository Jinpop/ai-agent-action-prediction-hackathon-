"""고전 ML 추가 짜내기: per-class 확률 가중치 최적화 + 스태킹 메타러너.

빠른 검증용(홀드아웃만). 기존 v7 블렌드 대비:
  (1) 단순 블렌드 (baseline)
  (2) per-class 가중치 최적화 (블렌드 proba 각 클래스 컬럼에 배율 -> macro-F1 최대)
  (3) 스태킹: train 을 A/B로 나눠 base(A)->proba(B)->meta LogReg(B), 홀드아웃 평가
속도 위해 어휘 축소(word30k+char20k), LogReg max_iter=600.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight

import feat

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RS = 42


def macro(y, p):
    return f1_score(y, p, average="macro")


print("Load data...")
samples = feat.load_jsonl(os.path.join(DATA, "train.jsonl"))
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
lm = dict(zip(labels["id"], labels["action"]))
ids = [s["id"] for s in samples]
y = np.array([lm[i] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])
texts = [feat.build_text(s) for s in samples]

print("Features...")
wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=5, max_features=30000,
                     sublinear_tf=True, dtype=np.float32)
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=20000,
                     sublinear_tf=True, dtype=np.float32)
Xs = sp_hstack([wv.fit_transform(texts), cv.fit_transform(texts)]).tocsr()
svd = TruncatedSVD(n_components=200, random_state=RS)
Xtext = svd.fit_transform(Xs).astype(np.float32)
meta = feat.build_meta_frame(samples).values.astype(np.float32)
Xhgb = np.hstack([Xtext, meta]).astype(np.float32)
ms = MaxAbsScaler().fit(meta)
Xlr = sp_hstack([Xs, csr_matrix(ms.transform(meta))]).tocsr()

tr, va = next(GroupKFold(n_splits=5).split(Xhgb, y, groups))


def fit_bases(idx):
    sw = compute_sample_weight("balanced", y[idx])
    h = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                       l2_regularization=1.0, random_state=RS).fit(
        Xhgb[idx], y[idx], sample_weight=sw)
    l = LogisticRegression(max_iter=600, C=3.0, class_weight="balanced").fit(Xlr[idx], y[idx])
    n = ComplementNB(alpha=0.3).fit(Xs[idx], y[idx])
    return h, l, n


def probs(models, i):
    h, l, n = models
    return h.predict_proba(Xhgb[i]), l.predict_proba(Xlr[i]), n.predict_proba(Xs[i])


print("Fit base on train...")
base = fit_bases(tr)
classes = list(base[0].classes_)
cls = np.array(classes)
ph, pl, pn = probs(base, va)

# (1) 단순 블렌드 w 탐색
best_w, best_f1 = (0.4, 0.45, 0.15), -1
for a in range(21):
    for b in range(21 - a):
        c = 20 - a - b
        w = (a / 20, b / 20, c / 20)
        f = macro(y[va], cls[(w[0]*ph+w[1]*pl+w[2]*pn).argmax(1)])
        if f > best_f1:
            best_f1, best_w = f, w
blend_va = best_w[0]*ph + best_w[1]*pl + best_w[2]*pn
print(f"(1) 단순 블렌드 w={tuple(round(x,2) for x in best_w)}  macro={best_f1:.4f}")

# (2) per-class 가중치 최적화 (coordinate ascent)
K = len(classes)
cw = np.ones(K)
cur = macro(y[va], cls[blend_va.argmax(1)])
for _ in range(6):
    for k in range(K):
        bestm, bestf = cw[k], cur
        for m in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0]:
            cw[k] = m
            f = macro(y[va], cls[(blend_va * cw).argmax(1)])
            if f > bestf:
                bestf, bestm = f, m
        cw[k] = bestm
        cur = bestf
print(f"(2) per-class 가중치 최적화  macro={cur:.4f}  (+{cur-best_f1:.4f})")

# (3) 스태킹: train 을 A/B 로 나눠 meta 학습
gk2 = GroupKFold(n_splits=2)
a_idx, b_idx = next(gk2.split(Xhgb[tr], y[tr], groups[tr]))
A, B = tr[a_idx], tr[b_idx]
baseA = fit_bases(A)
phB, plB, pnB = probs(baseA, B)
XB = np.hstack([phB, plB, pnB])
meta_lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(XB, y[B])
# 홀드아웃: base(train전체)로 proba -> meta 예측
Xva = np.hstack([ph, pl, pn])
stack_pred = meta_lr.predict(Xva)
print(f"(3) 스태킹 메타러너  macro={macro(y[va], stack_pred):.4f}")

print(f"\n요약: v7단순블렌드 {best_f1:.4f} | per-class {cur:.4f} | 스태킹 {macro(y[va], stack_pred):.4f}")
