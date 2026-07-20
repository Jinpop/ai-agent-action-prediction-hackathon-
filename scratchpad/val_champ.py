#!/usr/bin/env python3
"""Evaluate the v39 CHAMPION artifacts.pkl on MY holdout split, to locate the 0.6092 vs 0.6286 gap.
If champion scores ~0.6286 on my split -> my refit recipe is inferior (fix it).
If champion scores ~0.6092 on my split -> split differs; my A/B numbers are on a valid (harder) split.
"""
import sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np, joblib
sys.path.insert(0, "open/scripts")
import feat
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from scipy.sparse import hstack as sp_hstack, csr_matrix
import re as _re

_FILE_RE=_re.compile(r'[\w./-]+\.\w{1,5}'); _GLOB_RE=_re.compile(r'[*?]|\*\.\w+|\*\*')
REL_COLS=["rel_glob","rel_nfile","rel_wasread","rel_pastpath","rel_lastsib","rel_lastfail","rel_nopen","rel_dirword","rel_searchword"]
def _sr(x): return x if isinstance(x,str) else ("" if x is None else str(x))
def _relfeats(s):
    cp=_sr(s.get("current_prompt","")); cpl=cp.lower(); h=s.get("history") or []
    w=(s.get("session_meta") or {}).get("workspace") or {}; of=[_sr(f) for f in (w.get("open_files") or [])]
    past_reads=set(); past_paths=set(); last_result=""; last_action=None
    for e in h:
        if not isinstance(e,dict) or e.get("role")!="assistant_action": continue
        nm=e.get("name"); a=e.get("args") or {}; last_action=nm
        pth=_sr(a.get("path") or a.get("target") or "")
        if pth: past_paths.add(pth.lower())
        if nm=="read_file" and pth: past_reads.add(pth.lower())
        last_result=_sr(e.get("result_summary",""))
    return [1.0 if _GLOB_RE.search(cp) else 0.0, float(len(_FILE_RE.findall(cp))),
        1.0 if any(_sr(x).split("/")[-1].split(".")[0].lower() in cpl for x in past_reads) else 0.0,
        1.0 if any(_sr(x).split("/")[-1].split(".")[0].lower() in cpl for x in past_paths) else 0.0,
        1.0 if last_action in ("read_file","grep_search","glob_pattern","list_directory") else 0.0,
        1.0 if ("error" in last_result.lower() or "fail" in last_result.lower()) else 0.0, float(len(of)),
        1.0 if any(k in cpl for k in ["폴더","디렉","directory","folder","목록","list","ls "]) else 0.0,
        1.0 if any(k in cpl for k in ["찾","검색","어디","where","grep","search","usage","참조","정의"]) else 0.0]

def load_labels(p):
    d={};
    with open(p) as f:
        for row in csv.DictReader(f): d[row["id"]]=row["action"]
    return d
def src_text(s): return ("[SRC] "+("au" if str(s.get("id","")).startswith("sess_au") else "sim")+" ")+feat.build_text(s)

ACT=feat.ACTIONS; a2id={a:i for i,a in enumerate(ACT)}
real=feat.load_jsonl("open/data/train.jsonl"); lab=load_labels("open/data/train_labels.csv")
ids=[s["id"] for s in real]; y=np.array([a2id[lab[i]] for i in ids])
groups=np.array([feat.session_of(i) for i in ids])
tr,va=next(GroupKFold(5).split(ids,y,groups))
print(f"my holdout={len(va)}")

art=joblib.load("scratchpad/v39x/model/artifacts.pkl")
classes=list(art["classes_"])
sv=[real[i] for i in va]; yva=y[va]
texts=[src_text(s) for s in sv]
Xs=sp_hstack([v.transform(texts) for v in art["vectorizers"]]).tocsr()
Xtext=art["svd"].transform(Xs).astype(np.float32)
mb=feat.build_meta_frame(sv); rel=np.array([_relfeats(s) for s in sv],dtype=np.float32)
for j,c in enumerate(REL_COLS): mb[c]=rel[:,j]
Xmeta=mb.reindex(columns=art["meta_columns"],fill_value=0.0).values.astype(np.float32)
Xhgb=np.hstack([Xtext,Xmeta]).astype(np.float32)
Xlr=sp_hstack([Xs,csr_matrix(art["meta_scaler"].transform(Xmeta))]).tocsr()
p=(0.45*art["hgb"].predict_proba(Xhgb)+0.40*art["logreg"].predict_proba(Xlr)+0.15*art["nb"].predict_proba(Xs))
c2a=[classes.index(a) for a in ACT]; p=p[:,c2a]
f1=f1_score(yva,p.argmax(1),average="macro")
print(f"CHAMPION artifacts.pkl on MY holdout split: Macro-F1={f1:.4f}")
print("-> if ~0.6286: my refit recipe inferior. if ~0.6092: split differs, my A/B valid.")
