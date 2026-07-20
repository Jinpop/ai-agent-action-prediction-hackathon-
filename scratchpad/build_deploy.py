#!/usr/bin/env python3
"""Build DEPLOYABLE classic artifacts (v39-compatible structure) on ALL data.
Trains on 70000 real + mint variant (no holdout). Same recipe as refit_eval.
Outputs: scratchpad/artifacts_mint1.pkl (control), scratchpad/artifacts_full.pkl (experiment).
Structure keys must match v39 script.py loader: vectorizers, svd, meta_columns, meta_scaler,
hgb, logreg, nb, blend_w, classes_, rel_cols.
"""
import sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np, joblib
sys.path.insert(0, "open/scripts")
import feat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
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
    d={}
    with open(p) as f:
        for row in csv.DictReader(f): d[row["id"]]=row["action"]
    return d
def src_text(s): return ("[SRC] "+("au" if str(s.get("id","")).startswith("sess_au") else "sim")+" ")+feat.build_text(s)

def build_meta(samples, columns):
    base=feat.build_meta_frame(samples); rel=np.array([_relfeats(s) for s in samples],dtype=np.float32)
    for j,c in enumerate(REL_COLS): base[c]=rel[:,j]
    return base.reindex(columns=columns,fill_value=0.0).values.astype(np.float32)

ACT=feat.ACTIONS; a2id={a:i for i,a in enumerate(ACT)}
real=feat.load_jsonl("open/data/train.jsonl"); rlab=load_labels("open/data/train_labels.csv")
mlab=load_labels("open/data/train_mint_labels.csv")
META_COLS=list(joblib.load("scratchpad/v39x/model/artifacts.pkl")["meta_columns"])
# models trained on int labels (a2id=ACTIONS order) -> predict_proba cols follow ACTIONS.
# v39 script.py does classes.index(a) for a in ACTIONS -> classes_ must be ACTIONS strings (identity remap).
CLASSES=list(ACT)

def sqrt_w(yy):
    cnt=np.bincount(yy,minlength=len(ACT)).astype(np.float64); cnt[cnt==0]=1
    w=np.sqrt(cnt.sum()/cnt); w/=w.mean(); return w[yy]

VARIANTS={"mint1":"open/data/train_mint.jsonl","full":"open/data/train_mint_full.jsonl"}
# text identical across variants -> fit vecs/svd/nb once on mint1's texts (== full's texts)
real_ids=[s["id"] for s in real]; y_real=np.array([a2id[rlab[i]] for i in real_ids])
mint0=feat.load_jsonl(VARIANTS["mint1"]); mint_ids=[s["id"] for s in mint0]
y_mint=np.array([a2id[mlab[i]] for i in mint_ids])
y_all=np.concatenate([y_real,y_mint])
texts=[src_text(s) for s in (real+mint0)]
print("fit vecs (all data)...")
vecs=[TfidfVectorizer(analyzer="word",ngram_range=(1,2),max_features=50000,min_df=4,sublinear_tf=True),
      TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=40000,min_df=4,sublinear_tf=True)]
Xtxt=sp_hstack([v.fit_transform(texts) for v in vecs]).tocsr()
print("fit svd..."); svd=TruncatedSVD(n_components=200,random_state=0); Xsvd=svd.fit_transform(Xtxt).astype(np.float32)
print("fit nb..."); nb=ComplementNB(alpha=0.3).fit(Xtxt,y_all)
sw=sqrt_w(y_all)

for name,path in VARIANTS.items():
    print(f"=== {name} ===")
    samples=real+feat.load_jsonl(path)
    M=build_meta(samples,META_COLS)
    scaler=MaxAbsScaler().fit(M)
    print(" fit hgb..."); hgb=HistGradientBoostingClassifier(learning_rate=0.1,max_iter=500,max_leaf_nodes=63,l2_regularization=1.0,random_state=0)
    hgb.fit(np.hstack([Xsvd,M]),y_all,sample_weight=sw)
    print(" fit logreg..."); lr=LogisticRegression(C=3.0,max_iter=2000,class_weight="balanced")
    lr.fit(sp_hstack([Xtxt,csr_matrix(scaler.transform(M))]).tocsr(),y_all)
    art={"vectorizers":vecs,"svd":svd,"meta_columns":META_COLS,"meta_scaler":scaler,
         "hgb":hgb,"logreg":lr,"nb":nb,"blend_w":0.6,"classes_":np.array(CLASSES,dtype=object),"rel_cols":REL_COLS}
    out=f"scratchpad/artifacts_{name}.pkl"; joblib.dump(art,out); print(f" saved {out}")
print("DONE")
