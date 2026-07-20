"""v13 이종 스태킹 분석 — v12(풀 트랜스크립트) 확률 + 고전 ML 확률 결합.

[ 사전 준비 ]
  Kaggle 에서 받은 두 파일을 open/artifacts/hybrid/ 에 넣는다:
    - holdout_probs2.npy  (14000 x 14 로짓, ACTIONS 순서)
    - holdout_idx2.npy    (v12 홀드아웃 인덱스)

v11 분석(stack_hybrid.py)과 동일하되:
  - v12 분할이 v10 분할과 같으면 classic_holdout_probs.npz 캐시를 재사용(행 재정렬),
    다르면 v12 분할 기준으로 고전 3모델 재학습(~45분) 후 별도 캐시.
  - 최적 combiner 는 artifacts/stack_combiner2.pkl 로 저장.

실행: cd open && .venv/bin/python scripts/stack_hybrid2.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import feat
import train as classic

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ART = os.path.join(HERE, "..", "artifacts")
HYB = os.path.join(ART, "hybrid")
CACHE_V10 = os.path.join(ART, "classic_holdout_probs.npz")          # v10 분할 기준
IDX_V10 = os.path.join(HYB, "holdout_idx.npy")
CACHE_V12 = os.path.join(ART, "classic_holdout_probs_v12split.npz")  # v12 분할 기준


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


print("Load data...")
samples = feat.load_jsonl(os.path.join(DATA, "train.jsonl"))
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
label_map = dict(zip(labels["id"], labels["action"]))
ids = [s["id"] for s in samples]
y = np.array([label_map[i] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])

hyb_logits = np.load(os.path.join(HYB, "holdout_probs2.npy"))
va = np.load(os.path.join(HYB, "holdout_idx2.npy"))
tr = np.setdiff1d(np.arange(len(y)), va)
assert not (set(groups[tr]) & set(groups[va])), "세션이 tr/va 에 걸침!"
print(f"train={len(tr)}  holdout={len(va)}  (v12/Kaggle 분할 사용)")
p_hyb_actions = softmax(hyb_logits)

# ---------- 고전 ML 홀드아웃 확률 (캐시 재사용 or 재계산) ----------
ph = pl = pn = classes = None
if os.path.exists(CACHE_V10) and os.path.exists(IDX_V10):
    idx_v10 = np.load(IDX_V10)
    if set(idx_v10.tolist()) == set(va.tolist()):
        print("v10 분할과 동일 -> 캐시 재사용 (행 재정렬)")
        z = np.load(CACHE_V10, allow_pickle=True)
        pos = {int(i): k for k, i in enumerate(idx_v10)}
        order = np.array([pos[int(i)] for i in va])
        ph, pl, pn = z["ph"][order], z["pl"][order], z["pn"][order]
        classes = list(z["classes"])

if ph is None:
    if os.path.exists(CACHE_V12):
        print(f"Load cached: {CACHE_V12}")
        z = np.load(CACHE_V12, allow_pickle=True)
        idx_c = z["idx"]
        assert np.array_equal(idx_c, va), "v12 캐시 분할 불일치 — 캐시 삭제 후 재실행"
        ph, pl, pn, classes = z["ph"], z["pl"], z["pn"], list(z["classes"])
    else:
        print("Fit classic base models on v12-train 80% (~45분)...")
        texts = [feat.build_text(s) for s in samples]
        vecs = classic.make_vectorizers()
        tr_texts = [texts[i] for i in tr]
        for v in vecs:
            v.fit(tr_texts)
        Xs = classic.transform_text(vecs, texts)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import MaxAbsScaler
        svd = TruncatedSVD(n_components=classic.SVD_COMPONENTS,
                           random_state=classic.RANDOM_STATE)
        Xtext = svd.fit_transform(Xs[tr]).astype(np.float32)
        Xtext_va = svd.transform(Xs[va]).astype(np.float32)
        meta_df = feat.build_meta_frame(samples)
        Xmeta = meta_df.values.astype(np.float32)
        scaler = MaxAbsScaler().fit(Xmeta[tr])
        Xhgb_tr = np.hstack([Xtext, Xmeta[tr]])
        Xhgb_va = np.hstack([Xtext_va, Xmeta[va]])
        Xlr = sp_hstack([Xs, csr_matrix(scaler.transform(Xmeta))]).tocsr()
        sw = compute_sample_weight("balanced", y[tr])
        hgb = classic.make_hgb().fit(Xhgb_tr, y[tr], sample_weight=sw)
        lr = classic.make_logreg().fit(Xlr[tr], y[tr])
        nb = classic.make_nb().fit(Xs[tr], y[tr])
        classes = list(hgb.classes_)
        ph = hgb.predict_proba(Xhgb_va)
        pl = lr.predict_proba(Xlr[va])
        pn = nb.predict_proba(Xs[va])
        np.savez_compressed(CACHE_V12, ph=ph, pl=pl, pn=pn,
                            classes=np.array(classes), idx=va)
        print(f"cached -> {CACHE_V12}")

cls = np.array(classes)
a2c = [feat.ACTIONS.index(c) for c in classes]
p_hyb = p_hyb_actions[:, a2c]
y_va = y[va]

print(f"\n단독 성능:  HGB={f1_score(y_va, cls[ph.argmax(1)], average='macro'):.4f}"
      f"  LogReg={f1_score(y_va, cls[pl.argmax(1)], average='macro'):.4f}"
      f"  NB={f1_score(y_va, cls[pn.argmax(1)], average='macro'):.4f}"
      f"  v12={f1_score(y_va, cls[p_hyb.argmax(1)], average='macro'):.4f}")

# (a) 2-way
w3, _ = classic.search_blend3(y_va, ph, pl, pn, classes)
p_classic = w3[0] * ph + w3[1] * pl + w3[2] * pn
best_a, bw_a = -1, None
for w in np.arange(0, 1.0001, 0.05):
    f1 = f1_score(y_va, cls[(w * p_hyb + (1 - w) * p_classic).argmax(1)], average="macro")
    if f1 > best_a:
        best_a, bw_a = f1, w
print(f"\n(a) 2-way 블렌드: w_v12={bw_a:.2f}  ->  {best_a:.4f}")

# (b) 4-way
best_b, bw_b = -1, None
for a_ in range(11):
    for b_ in range(11 - a_):
        for c_ in range(11 - a_ - b_):
            d_ = 10 - a_ - b_ - c_
            w = (a_ / 10, b_ / 10, c_ / 10, d_ / 10)
            p = w[0] * ph + w[1] * pl + w[2] * pn + w[3] * p_hyb
            f1 = f1_score(y_va, cls[p.argmax(1)], average="macro")
            if f1 > best_b:
                best_b, bw_b = f1, w
print(f"(b) 4-way 블렌드: w(hgb,lr,nb,v12)={bw_b}  ->  {best_b:.4f}")

# (c) 스태킹 메타러너 (홀드아웃 내부 그룹 CV)
Xstack = np.hstack([ph, pl, pn, p_hyb])
g_va = groups[va]
meta_f1s, meta_preds = [], np.empty(len(va), dtype=object)
for mtr, mva in GroupKFold(n_splits=5).split(Xstack, y_va, g_va):
    m = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    m.fit(Xstack[mtr], y_va[mtr])
    meta_preds[mva] = m.predict(Xstack[mva])
    meta_f1s.append(f1_score(y_va[mva], meta_preds[mva], average="macro"))
best_c = f1_score(y_va, list(meta_preds), average="macro")
print(f"(c) 스태킹 메타러너(CV):  fold={[round(f,4) for f in meta_f1s]}  전체={best_c:.4f}")

results = {"a_2way": best_a, "b_4way": best_b, "c_stack": best_c}
name = max(results, key=results.get)
print(f"\n==> 최고 전략: {name}  Macro-F1={results[name]:.4f}"
      f"   (v11 스태킹 0.7006 / v12 단독 상단 참조)")

combiner = {"strategy": name, "classes": classes, "actions_order": feat.ACTIONS,
            "w3_classic": w3, "w_hybrid_2way": float(bw_a), "w4": bw_b}
if name == "c_stack":
    m = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    m.fit(Xstack, y_va)
    combiner["meta_model"] = m
joblib.dump(combiner, os.path.join(ART, "stack_combiner2.pkl"))
print(f"saved -> {os.path.join(ART, 'stack_combiner2.pkl')}")

best_p = {"a_2way": bw_a * p_hyb + (1 - bw_a) * p_classic,
          "b_4way": bw_b[0] * ph + bw_b[1] * pl + bw_b[2] * pn + bw_b[3] * p_hyb}.get(name)
if best_p is not None:
    print("\n[클래스별 — 최고 전략]")
    print(classification_report(y_va, cls[best_p.argmax(1)], digits=3, zero_division=0))
