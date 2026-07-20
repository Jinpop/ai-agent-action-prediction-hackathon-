#!/usr/bin/env python3
"""v43: champion-exact recipe + mint_full (3180, restored) + relv2. Weighting=balanced (confirmed in v42).
Texts AND meta both from mint2_full (6859 rows). Deploy artifacts, v39-compatible structure.
"""
import sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np, joblib
sys.path.insert(0, "open/scripts")
import feat
sys.path.insert(0,"scratchpad")
from relv2 import relfeats2, REL2_COLS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.utils.class_weight import compute_sample_weight
from scipy.sparse import hstack as sp_hstack, csr_matrix
import re as _re

_FILE_RE=_re.compile(r'[\w./-]+\.\w{1,5}'); _GLOB_RE=_re.compile(r'[*?]|\*\.\w+|\*\*')
REL_COLS=["rel_glob","rel_nfile","rel_wasread","rel_pastpath","rel_lastsib","rel_lastfail","rel_nopen","rel_dirword","rel_searchword"]
def _sr(x): return x if isinstance(x,str) else ("" if x is None else str(x))
def _relfeats(s):
    cp=_sr(s.get("current_prompt","")); cpl=cp.lower(); h=s.get("history") or []
    w=(s.get("session_meta") or {}).get("workspace") or {}; of=[_sr(f) for f in (w.get("open_files") or [])]
    pr=set(); pp=set(); lres=""; la=None
    for e in h:
        if not isinstance(e,dict) or e.get("role")!="assistant_action": continue
        nm=e.get("name"); a=e.get("args") or {}; la=nm; pth=_sr(a.get("path") or a.get("target") or "")
        if pth: pp.add(pth.lower())
        if nm=="read_file" and pth: pr.add(pth.lower())
        lres=_sr(e.get("result_summary",""))
    return [1.0 if _GLOB_RE.search(cp) else 0.0,float(len(_FILE_RE.findall(cp))),
        1.0 if any(_sr(x).split("/")[-1].split(".")[0].lower() in cpl for x in pr) else 0.0,
        1.0 if any(_sr(x).split("/")[-1].split(".")[0].lower() in cpl for x in pp) else 0.0,
        1.0 if la in ("read_file","grep_search","glob_pattern","list_directory") else 0.0,
        1.0 if ("error" in lres.lower() or "fail" in lres.lower()) else 0.0,float(len(of)),
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
    r2=np.array([relfeats2(s) for s in samples],dtype=np.float32)
    for j,c in enumerate(REL2_COLS): base[c]=r2[:,j]
    return base.reindex(columns=columns,fill_value=0.0).values.astype(np.float32)

ACT=feat.ACTIONS; a2id={a:i for i,a in enumerate(ACT)}
real=feat.load_jsonl("open/data/train.jsonl"); rlab=load_labels("open/data/train_labels.csv")
mlab=load_labels("open/data/train_mint_full_labels.csv")
META_COLS=list(joblib.load("scratchpad/v39x/model/artifacts.pkl")["meta_columns"])+REL2_COLS

real_ids=[s["id"] for s in real]; y_real=np.array([a2id[rlab[i]] for i in real_ids])
mint=feat.load_jsonl("open/data/train_mint_full.jsonl")
y_mint=np.array([a2id[mlab[s["id"]]] for s in mint])
y_all=np.concatenate([y_real,y_mint])
samples=real+mint
print(f"real={len(real)} mint2_full={len(mint)} total={len(samples)}", flush=True)
texts=[src_text(s) for s in samples]

print("fit vecs (champion-exact)...", flush=True)
vecs=[TfidfVectorizer(analyzer="word",ngram_range=(1,2),max_features=50000,min_df=4,max_df=0.9,
                      sublinear_tf=True,strip_accents="unicode",dtype=np.float32),
      TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=40000,min_df=4,
                      sublinear_tf=True,strip_accents="unicode",dtype=np.float32)]
Xtxt=sp_hstack([v.fit_transform(texts) for v in vecs]).tocsr()
print("fit svd (seed42)...", flush=True)
svd=TruncatedSVD(n_components=200,random_state=42); Xsvd=svd.fit_transform(Xtxt).astype(np.float32)
print("fit nb...", flush=True)
nb=ComplementNB(alpha=0.3).fit(Xtxt,y_all)
MF=build_meta(samples,META_COLS)
scaler=MaxAbsScaler().fit(MF)
sw=compute_sample_weight("balanced",y_all)
print("fit hgb (balanced, seed42)...", flush=True)
hgb=HistGradientBoostingClassifier(learning_rate=0.1,max_iter=500,max_leaf_nodes=63,
    l2_regularization=1.0,early_stopping=True,n_iter_no_change=20,random_state=42)
hgb.fit(np.hstack([Xsvd,MF]).astype(np.float32),y_all,sample_weight=sw)
print("fit logreg (balanced, seed42)...", flush=True)
lr=LogisticRegression(C=3.0,max_iter=2000,class_weight="balanced",random_state=42)
lr.fit(sp_hstack([Xtxt,csr_matrix(scaler.transform(MF))]).tocsr(),y_all)
art={"vectorizers":vecs,"svd":svd,"meta_columns":META_COLS,"meta_scaler":scaler,
     "hgb":hgb,"logreg":lr,"nb":nb,"blend_w":0.6,
     "classes_":np.array(list(ACT),dtype=object),"rel_cols":REL_COLS}
joblib.dump(art,"scratchpad/artifacts_v46.pkl",compress=3)
print("saved scratchpad/artifacts_v46.pkl", flush=True); print("DONE", flush=True)
