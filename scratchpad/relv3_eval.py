#!/usr/bin/env python3
"""relv3 A/B (§6.4): 128(기준) vs 132(+local 4피처 — depth 저상관 subset).
refit_eval3와 동일 split/레시피(챔피언-정확 파라미터·mint_full). 판정: delta가 +0.003 미만이면
재현 핸디캡(−0.0023, v41/v42)을 못 넘어 배포 비후보."""
import sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "open/scripts"); sys.path.insert(0, "scratchpad")
import feat
from relv2 import relfeats2, REL2_COLS as REL2_ALL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from scipy.sparse import hstack as sp_hstack, csr_matrix
import re as _re, joblib

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
    for row in csv.DictReader(open(p)): d[row["id"]]=row["action"]
    return d
def src_text(s): return ("[SRC] "+("au" if str(s.get("id","")).startswith("sess_au") else "sim")+" ")+feat.build_text(s)

ACT=feat.ACTIONS; a2id={a:i for i,a in enumerate(ACT)}
real=feat.load_jsonl("open/data/train.jsonl"); rlab=load_labels("open/data/train_labels.csv")
META128=list(joblib.load("scratchpad/v39x/model/artifacts.pkl")["meta_columns"])
REL3_COLS=["r2_prompt_file_edited","r2_zero_match_last","r2_prompt_overlap_prev","r2_last_num"]
META132=META128+REL3_COLS
real_ids=[s["id"] for s in real]; y_real=np.array([a2id[rlab[i]] for i in real_ids])
groups=np.array([feat.session_of(i) for i in real_ids])
tr0,va=next(GroupKFold(5).split(real_ids,y_real,groups))
va_sess=set(groups[va]); y_va=y_real[va]

mint=feat.load_jsonl("open/data/train_mint_full.jsonl"); mlab=load_labels("open/data/train_mint_full_labels.csv")
mids=[s["id"] for s in mint]; y_mint=np.array([a2id[mlab[i]] for i in mids])
g_mint=np.array([feat.session_of(i) for i in mids])
keep=np.array([k for k in range(len(mint)) if g_mint[k] not in va_sess])
samples=real+mint
texts=[src_text(s) for s in samples]
tr=np.concatenate([tr0,(len(real)+keep).astype(tr0.dtype)])
y_all=np.concatenate([y_real,y_mint])
print(f"train={len(tr)} holdout={len(va)}",flush=True)

vecs=[TfidfVectorizer(analyzer="word",ngram_range=(1,2),max_features=50000,min_df=4,max_df=0.9,
                      sublinear_tf=True,strip_accents="unicode",dtype=np.float32),
      TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=40000,min_df=4,
                      sublinear_tf=True,strip_accents="unicode",dtype=np.float32)]
print("vecs...",flush=True)
Xtr_txt=sp_hstack([v.fit_transform([texts[i] for i in tr]) for v in vecs]).tocsr()
Xva_txt=sp_hstack([v.transform([texts[i] for i in va]) for v in vecs]).tocsr()
print("svd...",flush=True)
svd=TruncatedSVD(200,random_state=42)
Xtr_svd=svd.fit_transform(Xtr_txt).astype(np.float32); Xva_svd=svd.transform(Xva_txt).astype(np.float32)
nb=ComplementNB(alpha=0.3).fit(Xtr_txt,y_all[tr]); p_nb=nb.predict_proba(Xva_txt)

base=feat.build_meta_frame(samples)
rel=np.array([_relfeats(s) for s in samples],dtype=np.float32)
for j,c in enumerate(REL_COLS): base[c]=rel[:,j]
r2=np.array([relfeats2(s) for s in samples],dtype=np.float32)
for j,c in enumerate(REL2_ALL): base[c]=r2[:,j]
print("relv2 nonzero frac:",float((r2!=0).mean()),flush=True)
sw=compute_sample_weight("balanced",y_all[tr])
def run(meta_cols,tag):
    M=base.reindex(columns=meta_cols,fill_value=0.0).values.astype(np.float32)
    Mtr,Mva=M[tr],M[va]
    scaler=MaxAbsScaler().fit(Mtr)
    hgb=HistGradientBoostingClassifier(learning_rate=0.1,max_iter=500,max_leaf_nodes=63,
        l2_regularization=1.0,early_stopping=True,n_iter_no_change=20,random_state=42)
    print(f"hgb {tag}...",flush=True)
    hgb.fit(np.hstack([Xtr_svd,Mtr]),y_all[tr],sample_weight=sw)
    p_hgb=hgb.predict_proba(np.hstack([Xva_svd,Mva]))
    print(f"logreg {tag}...",flush=True)
    lr=LogisticRegression(C=3.0,max_iter=2000,class_weight="balanced",random_state=42)
    lr.fit(sp_hstack([Xtr_txt,csr_matrix(scaler.transform(Mtr))]).tocsr(),y_all[tr])
    p_lr=lr.predict_proba(sp_hstack([Xva_txt,csr_matrix(scaler.transform(Mva))]).tocsr())
    def align(clf,p): return p[:,[list(clf.classes_).index(k) for k in range(len(ACT))]]
    blend=0.45*align(hgb,p_hgb)+0.40*align(lr,p_lr)+0.15*align(nb,p_nb)
    f1=f1_score(y_va,blend.argmax(1),average="macro")
    print(f"[{tag}] classic-blend holdout = {f1:.4f}",flush=True)
    return f1
f_base=run(META128,"128(기준 재현)")
f_r3=run(META132,"132(+local4)")
print(f"\nrelv3(local4) delta = {f_r3-f_base:+.4f}  (게이트: +0.003 미만이면 재현핸디캡 미달)",flush=True)
