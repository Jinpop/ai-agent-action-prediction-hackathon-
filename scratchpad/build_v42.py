#!/usr/bin/env python3
"""v42 rebuild: EXACT champion params + mint_full.
Champion-vs-mine diffs found: vec0 max_df=0.9, strip_accents='unicode' (both vecs),
dtype float32, HGB early_stopping=True/n_iter_no_change=20/random_state=42,
LogReg random_state=42, SVD random_state=42.
Unknown: HGB sample_weight. Resolve by determinism: fit HGB with sqrt vs balanced vs none
on mint1 (champion's data), compare predictions against champion HGB — the matching one
is what champion used. Then build final artifacts with mint_full using that weighting.
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
    return base.reindex(columns=columns,fill_value=0.0).values.astype(np.float32)

ACT=feat.ACTIONS; a2id={a:i for i,a in enumerate(ACT)}
real=feat.load_jsonl("open/data/train.jsonl"); rlab=load_labels("open/data/train_labels.csv")
mlab=load_labels("open/data/train_mint_labels.csv")
champ=joblib.load("scratchpad/v39x/model/artifacts.pkl")
META_COLS=list(champ["meta_columns"])

real_ids=[s["id"] for s in real]; y_real=np.array([a2id[rlab[i]] for i in real_ids])
mint1=feat.load_jsonl("open/data/train_mint.jsonl")
mintF=feat.load_jsonl("open/data/train_mint_full.jsonl")
y_mint=np.array([a2id[mlab[s["id"]]] for s in mint1])
y_all=np.concatenate([y_real,y_mint])
texts=[src_text(s) for s in (real+mint1)]  # texts identical for mint1/mintF

print("fit vecs (champion-exact params)...", flush=True)
vecs=[TfidfVectorizer(analyzer="word",ngram_range=(1,2),max_features=50000,min_df=4,max_df=0.9,
                      sublinear_tf=True,strip_accents="unicode",dtype=np.float32),
      TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=40000,min_df=4,
                      sublinear_tf=True,strip_accents="unicode",dtype=np.float32)]
Xtxt=sp_hstack([v.fit_transform(texts) for v in vecs]).tocsr()
print("fit svd (seed42)...", flush=True)
svd=TruncatedSVD(n_components=200,random_state=42); Xsvd=svd.fit_transform(Xtxt).astype(np.float32)
print("fit nb...", flush=True)
nb=ComplementNB(alpha=0.3).fit(Xtxt,y_all)

# ---- weight determination on mint1 (champion's own data) ----
M1=build_meta(real+mint1,META_COLS)
Xh1=np.hstack([Xsvd,M1]).astype(np.float32)
def sqrt_w(yy):
    cnt=np.bincount(yy,minlength=len(ACT)).astype(np.float64); cnt[cnt==0]=1
    w=np.sqrt(cnt.sum()/cnt); w/=w.mean(); return w[yy]
CANDS={"sqrt":sqrt_w(y_all),"balanced":compute_sample_weight("balanced",y_all),"none":None}
# compare on a fixed probe subset (first 3000 real rows)
probe=slice(0,3000)
champ_cols=[list(champ["classes_"]).index(a) for a in ACT]
# champion HGB probs on probe rows (champion feature space)
Xs_ch=sp_hstack([v.transform([texts[i] for i in range(3000)]) for v in champ["vectorizers"]]).tocsr()
Xh_ch=np.hstack([champ["svd"].transform(Xs_ch).astype(np.float32),M1[probe]]).astype(np.float32)
p_ch=champ["hgb"].predict_proba(Xh_ch)[:,champ_cols]
best=None
for wname,sw in CANDS.items():
    print(f"probe HGB [{wname}]...", flush=True)
    hg=HistGradientBoostingClassifier(learning_rate=0.1,max_iter=500,max_leaf_nodes=63,
        l2_regularization=1.0,early_stopping=True,n_iter_no_change=20,random_state=42)
    hg.fit(Xh1,y_all,sample_weight=sw)
    p=hg.predict_proba(Xh1[probe])
    agree=float((p.argmax(1)==p_ch.argmax(1)).mean()); mae=float(np.abs(p-p_ch).mean())
    print(f"  [{wname}] n_iter={hg.n_iter_} argmax-agree={agree:.4f} probMAE={mae:.5f}", flush=True)
    if best is None or agree>best[1]: best=(wname,agree,hg)
wname=best[0]; print(f"==> champion weighting inferred: {wname}", flush=True)

# ---- final v42 artifacts: mint_full + inferred weighting ----
MF=build_meta(real+mintF,META_COLS)
scaler=MaxAbsScaler().fit(MF)
print("fit final hgb (mint_full)...", flush=True)
hgb=HistGradientBoostingClassifier(learning_rate=0.1,max_iter=500,max_leaf_nodes=63,
    l2_regularization=1.0,early_stopping=True,n_iter_no_change=20,random_state=42)
hgb.fit(np.hstack([Xsvd,MF]).astype(np.float32),y_all,sample_weight=CANDS[wname])
print("fit final logreg (mint_full)...", flush=True)
lr=LogisticRegression(C=3.0,max_iter=2000,class_weight="balanced",random_state=42)
lr.fit(sp_hstack([Xtxt,csr_matrix(scaler.transform(MF))]).tocsr(),y_all)
art={"vectorizers":vecs,"svd":svd,"meta_columns":META_COLS,"meta_scaler":scaler,
     "hgb":hgb,"logreg":lr,"nb":nb,"blend_w":0.6,
     "classes_":np.array(list(ACT),dtype=object),"rel_cols":REL_COLS}
joblib.dump(art,"scratchpad/artifacts_v42.pkl",compress=3)
print("saved scratchpad/artifacts_v42.pkl  (weighting=%s)"%wname, flush=True)
print("DONE", flush=True)
