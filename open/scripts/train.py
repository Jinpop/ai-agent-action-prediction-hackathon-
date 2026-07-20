"""학습 스크립트 v3: HGB + LogisticRegression 확률 앙상블.

 - HGB    : [word+char TF-IDF -> SVD300] + meta/history 파생  (구조 신호에 강함)
 - LogReg : word+char TF-IDF sparse 전체 (SVD로 잃은 텍스트 신호 회복)
 - 최종   : blend = w*HGB_proba + (1-w)*LogReg_proba, w는 홀드아웃에서 튜닝
홀드아웃 단계에서 클래스별 F1(오류분석)도 출력한다.

산출물(work/model/artifacts.pkl):
  vectorizers, svd, meta_columns, hgb, logreg, blend_w, classes_
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import feat

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "artifacts", "model")
SVD_COMPONENTS = 300
RANDOM_STATE = 42


def make_vectorizers():
    return [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=4,
                        max_df=0.9, max_features=50000, dtype=np.float32,
                        sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4,
                        max_features=40000, dtype=np.float32,
                        sublinear_tf=True, strip_accents="unicode"),
    ]


def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.1, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )


def make_logreg():
    return LogisticRegression(
        max_iter=2000, C=3.0, class_weight="balanced",
        solver="lbfgs", random_state=RANDOM_STATE,
    )


def make_nb():
    return ComplementNB(alpha=0.3)


def search_blend3(y_va, ph, pl, pn, classes):
    """3-모델 확률 가중합의 macro-F1 최대 가중치 탐색 (step=0.05)."""
    cls = np.array(classes)
    best_w, best_f1 = (0.4, 0.4, 0.2), -1.0
    for a in range(21):
        for b in range(21 - a):
            c = 20 - a - b
            wa, wb, wc = a / 20, b / 20, c / 20
            pred = cls[(wa * ph + wb * pl + wc * pn).argmax(1)]
            f1 = f1_score(y_va, pred, average="macro")
            if f1 > best_f1:
                best_f1, best_w = f1, (wa, wb, wc)
    return best_w, best_f1


def transform_text(vectorizers, texts):
    return sp_hstack([v.transform(texts) for v in vectorizers]).tocsr()


def main():
    t0 = time.time()
    print("Load data...")
    samples = feat.load_jsonl(os.path.join(DATA, "train.jsonl"))
    labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
    label_map = dict(zip(labels["id"], labels["action"]))
    ids = [s["id"] for s in samples]
    y = np.array([label_map[i] for i in ids])
    groups = np.array([feat.session_of(i) for i in ids])
    print(f"  samples={len(samples)}  classes={len(set(y))}")

    print("Fit text vectorizers (word+char)...")
    texts = [feat.build_text(s) for s in samples]
    vectorizers = make_vectorizers()
    for v in vectorizers:
        v.fit(texts)
    Xs = transform_text(vectorizers, texts)          # sparse (LogReg 입력)
    print(f"  sparse dim={Xs.shape[1]}")

    print("Fit SVD (for HGB)...")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    Xtext = svd.fit_transform(Xs).astype(np.float32)

    print("Build meta features...")
    meta_df = feat.build_meta_frame(samples)
    meta_columns = list(meta_df.columns)
    Xmeta = meta_df.values.astype(np.float32)
    Xhgb = np.hstack([Xtext, Xmeta]).astype(np.float32)          # HGB 입력 (dense)
    meta_scaler = MaxAbsScaler().fit(Xmeta)
    Xmeta_s = csr_matrix(meta_scaler.transform(Xmeta))
    Xlr = sp_hstack([Xs, Xmeta_s]).tocsr()                       # LogReg 입력 (text+meta sparse)
    print(f"  HGB dim={Xhgb.shape[1]}  LogReg dim={Xlr.shape[1]}")

    # ---- 세션 단위 홀드아웃 ----
    print("Holdout validation (session-grouped, 20%)...")
    tr, va = next(GroupKFold(n_splits=5).split(Xhgb, y, groups))

    sw = compute_sample_weight("balanced", y[tr])
    hgb = make_hgb().fit(Xhgb[tr], y[tr], sample_weight=sw)
    lr = make_logreg().fit(Xlr[tr], y[tr])
    nb = make_nb().fit(Xs[tr], y[tr])
    assert list(hgb.classes_) == list(lr.classes_) == list(nb.classes_), "class 순서 불일치"
    classes = list(hgb.classes_)

    ph = hgb.predict_proba(Xhgb[va])
    pl = lr.predict_proba(Xlr[va])
    pn = nb.predict_proba(Xs[va])
    cls = np.array(classes)
    f1_hgb = f1_score(y[va], cls[ph.argmax(1)], average="macro")
    f1_lr = f1_score(y[va], cls[pl.argmax(1)], average="macro")
    f1_nb = f1_score(y[va], cls[pn.argmax(1)], average="macro")

    best_w, best_f1 = search_blend3(y[va], ph, pl, pn, classes)
    print(f"  HGB={f1_hgb:.4f}  LogReg={f1_lr:.4f}  NB={f1_nb:.4f}  "
          f"Blend3(w={tuple(round(x,2) for x in best_w)})={best_f1:.4f}")

    print("\n[클래스별 오류분석 — Blend3]")
    wa, wb, wc = best_w
    pred_best = cls[(wa * ph + wb * pl + wc * pn).argmax(1)]
    print(classification_report(y[va], pred_best, digits=3, zero_division=0))

    # 홀드아웃만 실험 시 재학습 생략 (빠른 반복용): python train.py --holdout-only
    if "--holdout-only" in sys.argv:
        wt = tuple(round(x, 2) for x in best_w)
        print(f"[holdout-only] refit 생략. blend_w={wt} best_f1={best_f1:.4f}")
        return

    # ---- 전체 재학습 (제출용) ----
    print("Refit on full data...")
    sw = compute_sample_weight("balanced", y)
    hgb_full = make_hgb().fit(Xhgb, y, sample_weight=sw)
    lr_full = make_logreg().fit(Xlr, y)
    nb_full = make_nb().fit(Xs, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "vectorizers": vectorizers, "svd": svd, "meta_columns": meta_columns,
        "meta_scaler": meta_scaler,
        "hgb": hgb_full, "logreg": lr_full, "nb": nb_full, "blend_w": best_w,
        "classes_": list(hgb_full.classes_), "holdout_f1": float(best_f1),
    }
    out = os.path.join(MODEL_DIR, "artifacts.pkl")
    joblib.dump(artifacts, out, compress=3)
    print(f"Saved {out}  ({os.path.getsize(out)/1e6:.1f} MB)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
