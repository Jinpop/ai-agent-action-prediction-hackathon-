"""고전 현대화 실험: TF-IDF를 풀 트랜스크립트([SRC]+[U]/[A]+[META]+[P])로 학습.

기존 classic(v7)은 build_text(프롬프트+최근 user 3턴+행동 토큰)만 사용 — 2세대 전 입력.
트랜스포머를 +0.05 올린 풀트랜스크립트를 고전에도 적용해 블렌드 멤버(0.4 비중) 업그레이드.
홀드아웃 판정 전용 (refit 없음). 비교 기준: classic v7 홀드아웃 0.6089.
"""
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
from train import make_vectorizers, make_hgb, make_logreg, make_nb, search_blend3, transform_text

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def build_transcript(s):
    src = "au" if _s(s.get("id", "")).startswith("sess_au") else "sim"
    parts = [f"[SRC] {src}"]
    for h in (s.get("history") or []):
        if not isinstance(h, dict):
            continue
        if h.get("role") == "user":
            parts.append("[U] " + _s(h.get("content", "")))
        elif h.get("role") == "assistant_action":
            a = h.get("args") or {}
            astr = " ".join(f"{k}={_s(v)}" for k, v in a.items())
            parts.append(f"[A] {_s(h.get('name'))} {astr} -> {_s(h.get('result_summary'))}")
    m = s.get("session_meta") or {}
    w = m.get("workspace") or {}
    parts.append(f"[META] tier={_s(m.get('user_tier'))} ci={_s(w.get('last_ci_status'))} "
                 f"dirty={int(bool(w.get('git_dirty')))} turn={m.get('turn_index', 0)} "
                 f"open={','.join(_s(p) for p in (w.get('open_files') or [])[:6])}")
    parts.append("[P] " + _s(s.get("current_prompt", "")))
    return "\n".join(parts)


t0 = time.time()
samples = feat.load_jsonl(os.path.join(DATA, "train.jsonl"))
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
lab = dict(zip(labels["id"], labels["action"]))
ids = [s["id"] for s in samples]
y = np.array([lab[i] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])
texts = [build_transcript(s) for s in samples]
meta_df = feat.build_meta_frame(samples)
Xmeta = meta_df.values.astype(np.float32)
print(f"[{time.time()-t0:.0f}s] 트랜스크립트 준비 (중앙길이 {int(np.median([len(t) for t in texts]))}자)", flush=True)

vecs = make_vectorizers()
for v in vecs:
    v.fit(texts)
Xs = transform_text(vecs, texts)
print(f"[{time.time()-t0:.0f}s] TF-IDF {Xs.shape}", flush=True)
svd = TruncatedSVD(n_components=200, random_state=42)
Xtext = svd.fit_transform(Xs).astype(np.float32)
Xhgb = np.hstack([Xtext, Xmeta]).astype(np.float32)
scaler = MaxAbsScaler().fit(Xmeta)
Xlr = sp_hstack([Xs, csr_matrix(scaler.transform(Xmeta))]).tocsr()
print(f"[{time.time()-t0:.0f}s] 피처 완료", flush=True)

tr, va = next(GroupKFold(5).split(Xhgb, y, groups))
sw = compute_sample_weight("balanced", y[tr])
hgb = make_hgb().fit(Xhgb[tr], y[tr], sample_weight=sw)
print(f"[{time.time()-t0:.0f}s] HGB 완료", flush=True)
lr = make_logreg().fit(Xlr[tr], y[tr])
nb = make_nb().fit(Xs[tr], y[tr])
classes = list(hgb.classes_)
cls = np.array(classes)
ph, pl, pn = hgb.predict_proba(Xhgb[va]), lr.predict_proba(Xlr[va]), nb.predict_proba(Xs[va])
print(f"[{time.time()-t0:.0f}s] HGB={f1_score(y[va],cls[ph.argmax(1)],average='macro'):.4f} "
      f"LogReg={f1_score(y[va],cls[pl.argmax(1)],average='macro'):.4f} "
      f"NB={f1_score(y[va],cls[pn.argmax(1)],average='macro'):.4f}", flush=True)
w, f = search_blend3(y[va], ph, pl, pn, classes)
print(f"[{time.time()-t0:.0f}s] Blend3 w={tuple(round(x,2) for x in w)} -> {f:.4f}  (v7 기준 0.6089)", flush=True)
p_blend = w[0]*ph + w[1]*pl + w[2]*pn
np.save(os.path.join(DATA, "..", "artifacts", "classic_v8_holdout_probs.npy"), p_blend)
np.save(os.path.join(DATA, "..", "artifacts", "classic_v8_holdout_idx.npy"), va)
import json
json.dump(classes, open(os.path.join(DATA, "..", "artifacts", "classic_v8_classes.json"), "w"))
print("저장 완료", flush=True)
