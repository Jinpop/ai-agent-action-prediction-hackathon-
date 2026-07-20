import warnings; warnings.filterwarnings('ignore')
import sys, csv, numpy as np, joblib
sys.path.insert(0,"open/scripts"); import feat
from sklearn.naive_bayes import ComplementNB
from scipy.sparse import hstack as sp_hstack
ch=joblib.load('scratchpad/v39x/model/artifacts.pkl')
def load_labels(p):
    d={}
    for row in csv.DictReader(open(p)): d[row["id"]]=row["action"]
    return d
ACT=feat.ACTIONS; a2id={a:i for i,a in enumerate(ACT)}
real=feat.load_jsonl("open/data/train.jsonl"); rlab=load_labels("open/data/train_labels.csv")
mint=feat.load_jsonl("open/data/train_mint.jsonl"); mlab=load_labels("open/data/train_mint_labels.csv")
y=np.array([a2id[rlab[s["id"]]] for s in real]+[a2id[mlab[s["id"]]] for s in mint])
base=[feat.build_text(s) for s in (real+mint)]
pref=[("[SRC] "+("au" if str(s.get("id","")).startswith("sess_au") else "sim")+" ")+t for s,t in zip(real+mint,base)]
v0,v1=ch['vectorizers']
cN=ch['nb'].feature_log_prob_
def test(tag, X):
    nb=ComplementNB(alpha=0.3).fit(X,y)
    D=np.abs(nb.feature_log_prob_-cN)
    print(f"[{tag}] maxdiff={D.max():.4f} mean={D.mean():.6f} frac<0.01={(D<0.01).mean():.4f}", flush=True)
    return D
D=test("with-SRC word|char", sp_hstack([v0.transform(pref),v1.transform(pref)]).tocsr())
print(f"   word블록 mean={D[:,:50000].mean():.6f}  char블록 mean={D[:,50000:].mean():.6f}", flush=True)
test("swap char|word", sp_hstack([v1.transform(pref),v0.transform(pref)]).tocsr())
test("no-SRC word|char", sp_hstack([v0.transform(base),v1.transform(base)]).tocsr())
print("DONE", flush=True)
