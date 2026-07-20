#!/usr/bin/env python3
"""Smoke-test a deploy artifacts.pkl through v39 script.py's EXACT classic path.
Validates: loads, feature dims align, classes remap works, probs finite & valid.
Usage: python smoke.py scratchpad/artifacts_full.pkl
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, joblib, pandas as pd
sys.path.insert(0, "open/scripts")
import feat
from scipy.sparse import hstack as sp_hstack, csr_matrix
import re as _re

PKL = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/artifacts_full.pkl"
_FILE_RE=_re.compile(r'[\w./-]+\.\w{1,5}'); _GLOB_RE=_re.compile(r'[*?]|\*\.\w+|\*\*')
REL_COLS=["rel_glob","rel_nfile","rel_wasread","rel_pastpath","rel_lastsib","rel_lastfail","rel_nopen","rel_dirword","rel_searchword"]
def _sr(x): return x if isinstance(x,str) else ("" if x is None else str(x))
def _relfeats(s):
    cp=_sr(s.get("current_prompt","")); cpl=cp.lower(); h=s.get("history") or []
    w=(s.get("session_meta") or {}).get("workspace") or {}; of=[_sr(f) for f in (w.get("open_files") or [])]
    pr=set(); pp=set(); lr=""; la=None
    for e in h:
        if not isinstance(e,dict) or e.get("role")!="assistant_action": continue
        nm=e.get("name"); a=e.get("args") or {}; la=nm; pth=_sr(a.get("path") or a.get("target") or "")
        if pth: pp.add(pth.lower())
        if nm=="read_file" and pth: pr.add(pth.lower())
        lr=_sr(e.get("result_summary",""))
    return [1.0 if _GLOB_RE.search(cp) else 0.0,float(len(_FILE_RE.findall(cp))),
        1.0 if any(_sr(x).split("/")[-1].split(".")[0].lower() in cpl for x in pr) else 0.0,
        1.0 if any(_sr(x).split("/")[-1].split(".")[0].lower() in cpl for x in pp) else 0.0,
        1.0 if la in ("read_file","grep_search","glob_pattern","list_directory") else 0.0,
        1.0 if ("error" in lr.lower() or "fail" in lr.lower()) else 0.0,float(len(of)),
        1.0 if any(k in cpl for k in ["폴더","디렉","directory","folder","목록","list","ls "]) else 0.0,
        1.0 if any(k in cpl for k in ["찾","검색","어디","where","grep","search","usage","참조","정의"]) else 0.0]

samples=feat.load_jsonl("open/data/test.jsonl")[:500]
n=len(samples)
meta_rows=pd.DataFrame([feat.build_meta_row(s) for s in samples])
_rel=[_relfeats(s) for s in samples]
for j,c in enumerate(REL_COLS): meta_rows[c]=[r[j] for r in _rel]
# ---- EXACT v39 classic block ----
art=joblib.load(PKL)
classes=list(art["classes_"])
texts_c=[("[SRC] "+("au" if str(s.get("id","")).startswith("sess_au") else "sim")+" ")+feat.build_text(s) for s in samples]
Xs=sp_hstack([v.transform(texts_c) for v in art["vectorizers"]]).tocsr()
Xtext=art["svd"].transform(Xs).astype(np.float32)
Xmeta=meta_rows.reindex(columns=art["meta_columns"],fill_value=0.0).values.astype(np.float32)
Xhgb=np.hstack([Xtext,Xmeta]).astype(np.float32)
Xlr=sp_hstack([Xs,csr_matrix(art["meta_scaler"].transform(Xmeta))]).tocsr()
p=(0.45*art["hgb"].predict_proba(Xhgb)+0.40*art["logreg"].predict_proba(Xlr)+0.15*art["nb"].predict_proba(Xs))
c2a=[classes.index(a) for a in feat.ACTIONS]
p=p[:,c2a]
ACT=np.array(feat.ACTIONS); preds=ACT[p.argmax(1)]
import collections
print(f"PKL={PKL}")
print(f"n={n}  meta_dim={Xmeta.shape[1]}  classes={len(classes)}")
print(f"probs finite={np.isfinite(p).all()}  rowsum~1={np.allclose(p.sum(1),1.0,atol=0.02)}  min={p.min():.4f} max={p.max():.4f}")
print(f"REL nonzero frac={np.mean(np.array(_rel)!=0):.3f}")
print(f"pred dist: {dict(collections.Counter(preds).most_common(6))}")
print(f"c2a identity? {c2a==list(range(14))}")
print("SMOKE OK" if (np.isfinite(p).all() and Xmeta.shape[1]==len(art['meta_columns'])) else "SMOKE FAIL")
