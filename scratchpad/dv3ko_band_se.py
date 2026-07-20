import sys, json, csv, collections, numpy as np
sys.path.insert(0, "open/scripts")
import feat
from sklearn.metrics import f1_score
ACT=list(feat.ACTIONS); a2id={a:i for i,a in enumerate(ACT)}
real=feat.load_jsonl("open/data/train.jsonl")
rlab={r["id"]:r["action"] for r in csv.DictReader(open("open/data/train_labels.csv"))}
oof=np.load("open/artifacts/oof/oof_classic_probs.npy")
def sm(z):
    z=z.astype(np.float64); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
M={"s48":("open/artifacts/hybrid/holdout_probs_s48.npy","open/artifacts/hybrid/holdout_idx_s48.npy"),
   "k79":("scratchpad/hp_kf768_79.npy","scratchpad/hidx.npy"),
   "k81":("scratchpad/hp_kf768_81.npy","scratchpad/hidx.npy"),
   "dv3ko_s79":("scratchpad/hp_dv3ko_s79.npy","scratchpad/hidx_dv3ko_s79.npy"),
   "dv3ko_s81":("scratchpad/hp_dv3ko_s81.npy","scratchpad/hidx_dv3ko_s81.npy")}
LOG={}; IDX={}
for k,(pp,ip) in M.items(): LOG[k]=np.load(pp); IDX[k]=np.load(ip)
hidx=np.load("scratchpad/hidx.npy")
inter=hidx  # all members share identical hidx order (verified)
def mprobs(k):
    pos_map={int(r):i for i,r in enumerate(IDX[k])}
    pos=np.array([pos_map[int(r)] for r in inter]); return sm(LOG[k][pos])
y=np.array([a2id[rlab[real[int(r)]["id"]]] for r in inter])
sess=np.array([feat.session_of(real[int(r)]["id"]) for r in inter])
def blendpred(tf):
    seed=np.mean([mprobs(k) for k in tf],axis=0)
    return (0.6*seed+0.4*oof[inter]).argmax(1)
combos={"v58-core":["s48","k79","k81"],
        "cand_dv79":["s48","k79","dv3ko_s79"],
        "cand_dv81":["s48","k79","dv3ko_s81"]}
preds={n:blendpred(tf) for n,tf in combos.items()}
# session-level bootstrap SE of plain macro-F1
usess=np.unique(sess); s2rows={s:np.where(sess==s)[0] for s in usess}
rng=np.random.default_rng(0); B=500
print("combo      plain_F1   bootSE(sess)")
boot={}
for n in combos:
    base_f=f1_score(y,preds[n],average="macro")
    fs=[]
    for _ in range(B):
        pick=rng.choice(usess,size=len(usess),replace=True)
        idx=np.concatenate([s2rows[s] for s in pick])
        fs.append(f1_score(y[idx],preds[n][idx],average="macro"))
    boot[n]=np.array(fs)
    print(f"{n:10s} {base_f:.4f}    {np.std(fs):.4f}")
# paired bootstrap of (cand - base) delta
print("\npaired delta vs v58-core (session bootstrap, same resample):")
for n in ["cand_dv79","cand_dv81"]:
    d=boot[n]-boot["v58-core"]
    pf=f1_score(y,preds[n],average="macro")-f1_score(y,preds["v58-core"],average="macro")
    print(f"  {n}: delta_plain={pf:+.4f}  bootSE(delta)={np.std(d):.4f}  P(delta<0)={np.mean(d<0):.2f}")
