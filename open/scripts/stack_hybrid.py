"""v11 이종 스태킹 분석 — 하이브리드(트랜스포머) 확률 + 고전 ML 확률 결합.

[ 사전 준비 ]
  Colab 에서 받은 두 파일을 open/artifacts/hybrid/ 에 넣는다:
    - holdout_probs.npy  (14000 x 14 로짓, ACTIONS 순서)
    - holdout_idx.npy    (홀드아웃 행 인덱스 — 분할 일치 검증용)

[ 하는 일 ]
  1) 고전 ML 3모델(HGB/LogReg/NB)을 train.py 와 동일하게 80% 학습 → 홀드아웃 확률
     (최초 1회 ~20분, artifacts/classic_holdout_probs.npz 에 캐시)
  2) 하이브리드 로짓 → softmax, 클래스 순서를 고전 ML(알파벳순)에 정렬
  3) 결합 전략 비교:
     (a) 2-way 블렌드  w*hybrid + (1-w)*classic_blend3   — w 그리드
     (b) 4-way 블렌드  (hgb, lr, nb, hybrid) 가중치 그리드
     (c) 스태킹 메타러너 (확률 56d -> LogReg), 홀드아웃 내부 세션-그룹 CV 로 공정 평가
  4) 최적 전략/가중치를 artifacts/stack_combiner.pkl 로 저장 (v11 포장용)

실행: cd open && .venv/bin/python scripts/stack_hybrid.py
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
import train as classic   # make_vectorizers/make_hgb/make_logreg/make_nb 재사용

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ART = os.path.join(HERE, "..", "artifacts")
HYB = os.path.join(ART, "hybrid")
CACHE = os.path.join(ART, "classic_holdout_probs.npz")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------- 데이터/분할 ----------
# 주의: Colab sklearn 과 로컬 sklearn 의 GroupKFold 배정이 달라 로컬 재현 불가.
# -> Colab 이 실제로 쓴 분할(holdout_idx.npy)을 그대로 사용해 정렬한다.
print("Load data...")
samples = feat.load_jsonl(os.path.join(DATA, "train.jsonl"))
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
label_map = dict(zip(labels["id"], labels["action"]))
ids = [s["id"] for s in samples]
y = np.array([label_map[i] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])

hyb_logits = np.load(os.path.join(HYB, "holdout_probs.npy"))
hyb_idx = np.load(os.path.join(HYB, "holdout_idx.npy"))
va = hyb_idx                                     # Colab 의 홀드아웃을 그대로 사용
tr = np.setdiff1d(np.arange(len(y)), va)
# 세션이 tr/va 에 걸치지 않는지 확인 (Colab 도 GroupKFold 였으므로 성립해야 함)
assert not (set(groups[tr]) & set(groups[va])), "세션이 tr/va 에 걸침!"
print(f"train={len(tr)}  holdout={len(va)}  (Colab 분할 사용)")

p_hyb_actions = softmax(hyb_logits)          # ACTIONS 순서, hyb_idx 행 순서

# ---------- 고전 ML 홀드아웃 확률 (캐시) ----------
if os.path.exists(CACHE):
    print(f"Load cached classic probs: {CACHE}")
    z = np.load(CACHE, allow_pickle=True)
    ph, pl, pn = z["ph"], z["pl"], z["pn"]
    classes = list(z["classes"])
else:
    print("Fit classic base models on 80% (최초 1회, ~20분)...")
    texts = [feat.build_text(s) for s in samples]
    vecs = classic.make_vectorizers()
    tr_texts = [texts[i] for i in tr]
    for v in vecs:
        v.fit(tr_texts)                      # 누수 방지: tr 로만 fit
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
    ph, pl, pn = (hgb.predict_proba(Xhgb_va), lr.predict_proba(Xlr[va]),
                  nb.predict_proba(Xs[va]))
    np.savez_compressed(CACHE, ph=ph, pl=pl, pn=pn, classes=np.array(classes))
    print(f"cached -> {CACHE}")

cls = np.array(classes)
# 하이브리드 확률을 고전 ML 클래스 순서로 정렬
a2c = [feat.ACTIONS.index(c) for c in classes]
p_hyb = p_hyb_actions[:, a2c]
y_va = y[va]

print(f"\n단독 성능:  HGB={f1_score(y_va, cls[ph.argmax(1)], average='macro'):.4f}"
      f"  LogReg={f1_score(y_va, cls[pl.argmax(1)], average='macro'):.4f}"
      f"  NB={f1_score(y_va, cls[pn.argmax(1)], average='macro'):.4f}"
      f"  Hybrid={f1_score(y_va, cls[p_hyb.argmax(1)], average='macro'):.4f}")

# ---------- (a) 2-way: hybrid + classic_blend3 ----------
w3, _ = classic.search_blend3(y_va, ph, pl, pn, classes)
p_classic = w3[0] * ph + w3[1] * pl + w3[2] * pn
best_a, bw_a = -1, None
for w in np.arange(0, 1.0001, 0.05):
    f1 = f1_score(y_va, cls[(w * p_hyb + (1 - w) * p_classic).argmax(1)], average="macro")
    if f1 > best_a:
        best_a, bw_a = f1, w
print(f"\n(a) 2-way 블렌드: w_hybrid={bw_a:.2f}  ->  {best_a:.4f}")

# ---------- (b) 4-way 그리드 (step 0.1) ----------
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
print(f"(b) 4-way 블렌드: w(hgb,lr,nb,hyb)={bw_b}  ->  {best_b:.4f}")

# ---------- (c) 스태킹 메타러너 (홀드아웃 내부 그룹 CV 로 공정 평가) ----------
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

# ---------- 최적 전략 저장 ----------
results = {"a_2way": (best_a, {"w_hybrid": float(bw_a), "w3_classic": w3}),
           "b_4way": (best_b, {"w4": bw_b}),
           "c_stack": (best_c, None)}
name = max(results, key=lambda k: results[k][0])
print(f"\n==> 최고 전략: {name}  Macro-F1={results[name][0]:.4f}"
      f"   (v10 단독 0.6519 / v7 0.6089)")

combiner = {"strategy": name, "classes": classes, "actions_order": feat.ACTIONS,
            "w3_classic": w3, "w_hybrid_2way": float(bw_a), "w4": bw_b}
if name == "c_stack":
    m = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    m.fit(Xstack, y_va)                      # 제출용: 홀드아웃 전체로 메타러너 학습
    combiner["meta_model"] = m
joblib.dump(combiner, os.path.join(ART, "stack_combiner.pkl"))
print(f"saved -> {os.path.join(ART, 'stack_combiner.pkl')}")

best_p = {"a_2way": bw_a * p_hyb + (1 - bw_a) * p_classic,
          "b_4way": bw_b[0] * ph + bw_b[1] * pl + bw_b[2] * pn + bw_b[3] * p_hyb}.get(name)
if best_p is not None:
    print("\n[클래스별 — 최고 전략]")
    print(classification_report(y_va, cls[best_p.argmax(1)], digits=3, zero_division=0))
