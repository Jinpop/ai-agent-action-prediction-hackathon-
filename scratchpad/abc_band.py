import sys, json, csv, collections, numpy as np
sys.path.insert(0, "open/scripts")
import feat
from sklearn.metrics import f1_score

ACT = list(feat.ACTIONS); a2id = {a:i for i,a in enumerate(ACT)}
real = feat.load_jsonl("open/data/train.jsonl")
rlab = {r["id"]: r["action"] for r in csv.DictReader(open("open/data/train_labels.csv"))}

# classic OOF (probs, ACTIONS order, jsonl row order). Verify id alignment.
oof = np.load("open/artifacts/oof/oof_classic_probs.npy")           # (70000,14) probs
oof_ids = json.load(open("open/artifacts/oof/oof_classic_ids.json"))
assert oof.shape==(70000,14)
for r in [0, 123, 45678, 69999]:
    assert oof_ids[r]==real[r]["id"], f"oof id mismatch @ {r}"
assert abs(oof.sum(1).mean()-1)<1e-4 and (oof>=0).all()   # already probs

def sm(z):
    z=z.astype(np.float64); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

# transformer members: (logits_path, idx_path). All logits -> softmax.
M = {
 "s48":  ("open/artifacts/hybrid/holdout_probs_s48.npy","open/artifacts/hybrid/holdout_idx_s48.npy"),
 "k79":  ("scratchpad/hp_kf768_79.npy","scratchpad/hidx.npy"),
 "k81":  ("scratchpad/hp_kf768_81.npy","scratchpad/hidx.npy"),
 "k78":  ("scratchpad/hp_kf768_78.npy","scratchpad/hidx.npy"),
 "pt74c":("scratchpad/hp_kf768pt74c.npy","scratchpad/hidx_pt74c.npy"),
 "h83":  ("scratchpad/hv_h83_probs.npy","scratchpad/hv_h83_idx.npy"),
}
LOG={}; IDX={}
for k,(pp,ip) in M.items():
    LOG[k]=np.load(pp); IDX[k]=np.load(ip)
    assert LOG[k].shape==(14000,14) and IDX[k].shape==(14000,)

def member_probs_on(k, inter):
    pos_map={int(r):i for i,r in enumerate(IDX[k])}
    pos=np.array([pos_map[int(r)] for r in inter])
    return sm(LOG[k][pos])            # per-member softmax on inter rows

def blend_f1(tf_members, inter):
    seed=np.mean([member_probs_on(k,inter) for k in tf_members],axis=0)  # mean per-member softmax
    p_classic=oof[inter]                                                 # already probs, ACTIONS order
    blend=0.6*seed+0.4*p_classic
    pred=blend.argmax(1)
    y=np.array([a2id[rlab[real[int(r)]["id"]]] for r in inter])
    g=[feat.session_of(real[int(r)]["id"]) for r in inter]
    cnt=collections.Counter(g); w=np.array([1.0/cnt[x] for x in g])
    return (f1_score(y,pred,average="macro"),
            f1_score(y,pred,average="macro",sample_weight=w))

def inter_of(tf_members):
    s=None
    for k in tf_members:
        v=set(int(x) for x in IDX[k])
        s=v if s is None else (s & v)
    return np.array(sorted(s))

BASE=["s48","k79","k81"]
combos={"A (s48->k78)":["k79","k81","k78"],
        "B (s48->h83)":["k79","k81","h83"],
        "C (s48->pt74c)":["k79","k81","pt74c"]}

# sanity vs checklist: v58-core full-blend on full hidx (14000)
inter_full=inter_of(BASE)
bf_plain,bf_se=blend_f1(BASE,inter_full)
print(f"[SANITY] v58-core full-blend on hidx n={len(inter_full)}: plain={bf_plain:.4f} sess-eq={bf_se:.4f} (checklist ref champion-core 0.7688)")
# sanity vs band_check_sanity: 3TF-only sess-eq on hidx∩h100_va
va=set(int(x) for x in IDX['h83']); inter_bva=np.array(sorted(set(int(x) for x in IDX['s48']) & va))
seed3=np.mean([member_probs_on(k,inter_bva) for k in BASE],axis=0)
yb=np.array([a2id[rlab[real[int(r)]["id"]]] for r in inter_bva])
gb=[feat.session_of(real[int(r)]["id"]) for r in inter_bva]; cb=collections.Counter(gb); wb=np.array([1.0/cb[x] for x in gb])
print(f"[SANITY] 3TF-only(s48+k79+k81) sess-eq on hidx∩va n={len(inter_bva)}: {f1_score(yb,seed3.argmax(1),average='macro',sample_weight=wb):.4f} (reproduces band_check_sanity b58 shape)")

print("\n=== A/B/C BAND CHECK (full deployment blend 0.4*classic+0.6*mean(3TF)) ===")
print(f"{'combo':16s} {'n_rows':>6s} {'n_sess':>6s} {'base_plain':>10s} {'cand_plain':>10s} {'gate(-.0058)':>12s} {'inband(plain)':>13s} | {'base_se':>7s} {'cand_se':>7s} {'inband(se)':>10s}")
for name,tf in combos.items():
    inter=inter_of(tf)
    nsess=len(set(feat.session_of(real[int(r)]["id"]) for r in inter))
    bp,bs=blend_f1(BASE,inter)      # matched v58-core baseline on SAME split
    cp,cs=blend_f1(tf,inter)        # candidate
    gate_p=bp-0.0058; gate_s=bs-0.0058
    ib_p="Y" if cp>=gate_p else "N"; ib_s="Y" if cs>=gate_s else "N"
    print(f"{name:16s} {len(inter):6d} {nsess:6d} {bp:10.4f} {cp:10.4f} {gate_p:12.4f} {ib_p:>13s} | {bs:7.4f} {cs:7.4f} {ib_s:>10s}")
