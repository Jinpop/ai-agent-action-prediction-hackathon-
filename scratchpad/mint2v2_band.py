import json, csv, collections, numpy as np, sys
sys.path.insert(0, "open/scripts")
import feat
from sklearn.metrics import f1_score

ACT = list(feat.ACTIONS); a2id = {a:i for i,a in enumerate(ACT)}
real = feat.load_jsonl("open/data/train.jsonl")
rlab = {r["id"]: r["action"] for r in csv.DictReader(open("open/data/train_labels.csv"))}

# classic OOF (probs, ACTIONS order, jsonl row order) — NOT re-softmaxed
oof = np.load("open/artifacts/oof/oof_classic_probs.npy")
oof_ids = json.load(open("open/artifacts/oof/oof_classic_ids.json"))
assert oof.shape==(70000,14)
for r in [0,123,45678,69999]:
    assert oof_ids[r]==real[r]["id"], f"oof id mismatch @ {r}"
assert abs(oof.sum(1).mean()-1)<1e-4 and (oof>=0).all()
print(f"[CONTRACT] classic OOF rowsum mean={oof.sum(1).mean():.6f} (already probs, NOT re-softmaxed)")

def sm(z):
    z=z.astype(np.float64); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

M = {
 "s48":         ("open/artifacts/hybrid/holdout_probs_s48.npy","open/artifacts/hybrid/holdout_idx_s48.npy"),
 "k81":         ("scratchpad/hp_kf768_81.npy","scratchpad/hidx.npy"),
 "mint2b_s79":  ("scratchpad/hp_mint2b_s79.npy","scratchpad/hidx.npy"),
 "mint2v2_s79": ("scratchpad/harvest_mint2v2/s79/holdout_probs3.npy","scratchpad/harvest_mint2v2/s79/holdout_idx3.npy"),
 "mint2v2_s81": ("scratchpad/harvest_mint2v2/s81/holdout_probs3.npy","scratchpad/harvest_mint2v2/s81/holdout_idx3.npy"),
}
LOG={}; IDX={}
for k,(pp,ip) in M.items():
    LOG[k]=np.load(pp); IDX[k]=np.load(ip)
    assert LOG[k].shape==(14000,14) and IDX[k].shape==(14000,), f"{k} shape {LOG[k].shape} {IDX[k].shape}"

# ---- per-member softmax necessity ----
print("\n[CONTRACT] per-member raw-logits check (rowsum!=1 => raw logits => softmax applied):")
softmax_flag={}
for k in M:
    rs=LOG[k].sum(1); is_prob=(abs(rs.mean()-1)<1e-3) and (LOG[k].min()>=-1e-6)
    softmax_flag[k]= not is_prob
    print(f"  {k:12s}: rowsum mean={rs.mean():.4f} min={LOG[k].min():.3f} max={LOG[k].max():.3f} -> {'PROBS(no sm)' if is_prob else 'RAW LOGITS (softmax applied)'}")

# ---- new-member idx == hidx ----
hidx=np.load("scratchpad/hidx.npy")
split={}
for nk in ["mint2v2_s79","mint2v2_s81"]:
    so=bool(np.array_equal(IDX[nk],hidx)); ss=set(int(x) for x in IDX[nk])==set(int(x) for x in hidx)
    split[nk]={"same_order":so,"same_set":ss}
    print(f"[SPLIT] {nk} idx3 == hidx  same_order={so}  same_set={ss}")

def member_probs_on(k, inter):
    pos_map={int(r):i for i,r in enumerate(IDX[k])}
    pos=np.array([pos_map[int(r)] for r in inter])
    return sm(LOG[k][pos])

def blend_f1(tf_members, inter):
    seed=np.mean([member_probs_on(k,inter) for k in tf_members],axis=0)
    blend=0.6*seed+0.4*oof[inter]
    pred=blend.argmax(1)
    y=np.array([a2id[rlab[real[int(r)]["id"]]] for r in inter])
    g=[feat.session_of(real[int(r)]["id"]) for r in inter]
    cnt=collections.Counter(g); w=np.array([1.0/cnt[x] for x in g])
    return (f1_score(y,pred,average="macro"),
            f1_score(y,pred,average="macro",sample_weight=w))

def inter_of(tf_members):
    s=None
    for k in tf_members:
        v=set(int(x) for x in IDX[k]); s=v if s is None else (s&v)
    return np.array(sorted(s))

CHAMP=["s48","k81","mint2b_s79"]
PROBES={"mint2v2_s79":["s48","k81","mint2v2_s79"],
        "mint2v2_s81":["s48","k81","mint2v2_s81"]}

BAND=0.0058
results={}
for pk,PROBE in PROBES.items():
    inter=inter_of(CHAMP+[pk])
    nsess=len(set(feat.session_of(real[int(r)]["id"]) for r in inter))
    cp_p,cp_s=blend_f1(CHAMP,inter)
    pr_p,pr_s=blend_f1(PROBE,inter)
    lo_p=cp_p-BAND; lo_s=cp_s-BAND
    ibp="Y" if pr_p>=lo_p else "N"; ibs="Y" if pr_s>=lo_s else "N"
    results[pk]={"n_rows":int(len(inter)),"n_sess":int(nsess),
                 "champ_plain":cp_p,"champ_sesseq":cp_s,
                 "probe_plain":pr_p,"probe_sesseq":pr_s,
                 "band_lower_plain":lo_p,"band_lower_sesseq":lo_s,
                 "delta_plain":pr_p-cp_p,"delta_sesseq":pr_s-cp_s,
                 "in_band_plain":ibp,"in_band_sesseq":ibs,
                 "same_order":split[pk]["same_order"],"same_set":split[pk]["same_set"]}
    print(f"\n=== PROBE {pk}  (CHAMP={CHAMP} -> {PROBE}) ===")
    print(f"  n_rows={len(inter)} n_sess={nsess}")
    print(f"  CHAMP(cA) reproduce   plain={cp_p:.6f}  sess-eq={cp_s:.6f}")
    print(f"  PROBE {pk:11s}     plain={pr_p:.6f}  sess-eq={pr_s:.6f}")
    print(f"  band lower(champ-0.0058) plain={lo_p:.6f} sess-eq={lo_s:.6f}")
    print(f"  delta(probe-champ)    plain={pr_p-cp_p:+.6f}  sess-eq={pr_s-cp_s:+.6f}")
    print(f"  IN-BAND (plain)={ibp}   IN-BAND (sess-eq)={ibs}")

json.dump(results, open("scratchpad/mint2v2_band_results.json","w"), indent=2, default=float)
json.dump({"softmax_flag":softmax_flag,"split":split}, open("scratchpad/mint2v2_band_contract.json","w"), indent=2, default=str)
print("\n[WROTE] scratchpad/mint2v2_band_results.json")
