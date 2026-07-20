"""baseline 을 우리와 '같은 세션 홀드아웃 split' 으로 자체평가.

(1) 제공된 baseline pkl 을 홀드아웃에 예측 -> 누수 있음(참고용 상한).
(2) baseline 방식(current_prompt 만 TF-IDF+LogReg)을 80% 로 재학습 -> 20% 평가 (공정).
우리 모델(train.py)과 동일한 GroupKFold(5).split 첫 분할을 사용한다.
"""
import os
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

import feat

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BASE_ZIP = os.path.join(HERE, "..", "baseline_submit.zip")

print("Load data...")
samples = feat.load_jsonl(os.path.join(DATA, "train.jsonl"))
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
label_map = dict(zip(labels["id"], labels["action"]))
ids = [s["id"] for s in samples]
y = np.array([label_map[i] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])
texts = [feat.extract_current_prompt(s) for s in samples]  # current_prompt 만

# 우리 train.py 와 동일한 첫 홀드아웃 분할
tr, va = next(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
print(f"  holdout val n={len(va)}")

# (1) 제공된 baseline pkl (누수 있음)
print("\n[1] 제공된 baseline pkl (누수 있음, 참고 상한)")
with zipfile.ZipFile(BASE_ZIP) as z:
    with z.open("model/tfidf_logreg.pkl") as f:
        base_model = joblib.load(f)
pred_full = base_model.predict([texts[i] for i in va])
f1_leak = f1_score(y[va], pred_full, average="macro")
print(f"  Macro-F1(holdout, leak) = {f1_leak:.4f}")

# (2) baseline 방식 재학습 (공정)
print("\n[2] baseline 방식 재학습 (current_prompt only, 80%->20%, 공정)")
pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
    LogisticRegression(max_iter=1000, n_jobs=-1),
)
pipe.fit([texts[i] for i in tr], y[tr])
pred_fair = pipe.predict([texts[i] for i in va])
f1_fair = f1_score(y[va], pred_fair, average="macro")
print(f"  Macro-F1(holdout, fair) = {f1_fair:.4f}")

print("\n=== 요약 (같은 홀드아웃 기준) ===")
print(f"  baseline pkl (누수)     : {f1_leak:.4f}")
print(f"  baseline 방식 (공정)    : {f1_fair:.4f}")
print(f"  우리 v1                 : 0.5245")
print(f"  우리 v2                 : 0.5526")
