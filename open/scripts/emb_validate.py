"""가설 검증: 다국어 문장 임베딩을 추가하면 탐색 4형제 F1이 오르는가?

subset(기본 24k)으로 빠르게:
  A) [SVD-TFIDF + meta]            (현재 방식)
  B) [SVD-TFIDF + meta + 임베딩]   (임베딩 추가)
를 HGB / LogReg 각각으로 세션 홀드아웃 비교. 전체 macro-F1 + 4형제 F1 출력.
임베딩은 emb_cache.npy 로 캐시.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight

import feat

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
FILE4 = ["read_file", "grep_search", "glob_pattern", "list_directory"]


def macro(y, p):
    return f1_score(y, p, average="macro")


def cluster_f1(y, p):
    return {c: f1_score(y == c, p == c) for c in FILE4}


def main():
    print(f"Load {N} samples...")
    samples, ids = [], []
    with open(os.path.join(DATA, "train.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N:
                break
            r = pd.io.json.loads(line) if False else __import__("json").loads(line)
            samples.append(r); ids.append(r["id"])
    labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
    lm = dict(zip(labels["id"], labels["action"]))
    y = np.array([lm[i] for i in ids])
    groups = np.array([feat.session_of(i) for i in ids])
    texts = [feat.build_text(s) for s in samples]

    # ---- 임베딩 (캐시) ----
    cache = os.path.join(HERE, "..", "artifacts", f"emb_cache_{N}.npy")
    if os.path.exists(cache):
        print("Load cached embeddings...")
        emb = np.load(cache)
    else:
        print(f"Embedding with {MODEL_NAME} (CPU, 시간 걸림)...")
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(MODEL_NAME)
        emb = st.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)
        np.save(cache, emb)
    print(f"  emb shape={emb.shape}")

    # ---- 공통 피처 ----
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=4,
                           max_features=40000, sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4,
                           max_features=30000, sublinear_tf=True, dtype=np.float32)
    Xs = sp_hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    svd = TruncatedSVD(n_components=200, random_state=42)
    Xtext = svd.fit_transform(Xs).astype(np.float32)
    meta = feat.build_meta_frame(samples).values.astype(np.float32)

    tr, va = next(GroupKFold(n_splits=5).split(Xtext, y, groups))
    sw = compute_sample_weight("balanced", y[tr])

    def run_hgb(X):
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.12,
                                           max_leaf_nodes=63, random_state=42)
        m.fit(X[tr], y[tr], sample_weight=sw)
        return m.predict(X[va])

    def run_lr(Xsp):
        m = LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced")
        m.fit(Xsp[tr], y[tr])
        return m.predict(Xsp[va])

    A_hgb = np.hstack([Xtext, meta])
    B_hgb = np.hstack([Xtext, meta, emb])
    ms = MaxAbsScaler().fit(meta)
    es = MaxAbsScaler().fit(emb)
    A_lr = sp_hstack([Xs, csr_matrix(ms.transform(meta))]).tocsr()
    B_lr = sp_hstack([Xs, csr_matrix(ms.transform(meta)),
                      csr_matrix(es.transform(emb))]).tocsr()

    print("\n=== HGB ===")
    for name, X in [("A(no emb)", A_hgb), ("B(+emb)", B_hgb)]:
        p = run_hgb(X)
        cf = cluster_f1(y[va], p)
        print(f"  {name}: macro={macro(y[va],p):.4f}  4형제={{" +
              ", ".join(f'{k[:4]}:{v:.3f}' for k, v in cf.items()) + "}")

    print("\n=== LogReg ===")
    for name, X in [("A(no emb)", A_lr), ("B(+emb)", B_lr)]:
        p = run_lr(X)
        cf = cluster_f1(y[va], p)
        print(f"  {name}: macro={macro(y[va],p):.4f}  4형제={{" +
              ", ".join(f'{k[:4]}:{v:.3f}' for k, v in cf.items()) + "}")


if __name__ == "__main__":
    main()
