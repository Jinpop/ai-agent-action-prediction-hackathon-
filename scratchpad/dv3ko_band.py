import sys, json, csv, collections, numpy as np
sys.path.insert(0, "open/scripts")
import feat
from sklearn.metrics import f1_score

ACT = list(feat.ACTIONS); a2id = {a:i for i,a in enumerate(ACT)}
real = feat.load_jsonl("open/data/train.jsonl")
rlab = {r["id"]: r["action"] for r in csv.DictReader(open("open/data/train_labels.csv"))}

# classic OOF (probs, ACTIONS order, jsonl row order)
oof = np.load("open/artifacts/oof/oof_classic_probs.npy")
oof_ids = json.load(open("open/artifacts/oof/oof_classic_ids.json"))
assert oof.shape==(70000,14)
for r in [0, 123, 45678, 69999]:
    assert oof_ids[r]==real[r]["id"], f"oof id mismatch @ {r}"
assert abs(oof.sum(1).mean()-1)<1e-4 and (oof>=0).all()
print(f"[CONTRACT] classic OOF: sum/row mean={oof.sum(1).mean():.6f} (already probs, NOT re-softmaxed)")

def sm(z):
    z=z.astype(np.float64); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

# transformer members: (logits_path, idx_path). All raw logits -> per-member softmax.
M = {
 "s48":        ("open/artifacts/hybrid/holdout_probs_s48.npy","open/artifacts/hybrid/holdout_idx_s48.npy"),
 "k79":        ("scratchpad/hp_kf768_79.npy","scratchpad/hidx.npy"),
 "k81":        ("scratchpad/hp_kf768_81.npy","scratchpad/hidx.npy"),
 "dv3ko_s79":  ("scratchpad/hp_dv3ko_s79.npy","scratchpad/hidx_dv3ko_s79.npy"),
 "dv3ko_s81":  ("scratchpad/hp_dv3ko_s81.npy","scratchpad/hidx_dv3ko_s81.npy"),
}
LOG={}; IDX={}
for k,(pp,ip) in M.items():
    LOG[k]=np.load(pp); IDX[k]=np.load(ip)
    assert LOG[k].shape==(14000,14) and IDX[k].shape==(14000,), f"{k} shape {LOG[k].shape} {IDX[k].shape}"

# --- CONTRACT: confirm raw-logits vs probs for each member (softmax necessity) ---
print("\n[CONTRACT] per-member raw-logits check (rowsum!=1 => raw logits => softmax applied):")
for k in M:
    rs = LOG[k].sum(1); is_prob = (abs(rs.mean()-1)<1e-3) and (LOG[k].min()>=-1e-6)
    print(f"  {k:11s}: rowsum mean={rs.mean():.4f} min={LOG[k].min():.3f} max={LOG[k].max():.3f} -> {'PROBS(no softmax)' if is_prob else 'RAW LOGITS (softmax applied)'}")

# --- dv3ko holdout_idx == hidx check (task-required) ---
hidx = np.load("scratchpad/hidx.npy")
print("\n[SPLIT] dv3ko holdout_idx3 vs local hidx.npy (must match: HOLDOUT_IDX=hidx pin):")
for k in ["dv3ko_s79","dv3ko_s81"]:
    same_order = np.array_equal(IDX[k], hidx)
    same_set = set(int(x) for x in IDX[k])==set(int(x) for x in hidx)
    print(f"  {k}: idx==hidx same_order={same_order} same_set={same_set}")

# --- ALIGNMENT VERIFY: single-member holdout F1 (uses each member's own idx order) ---
print("\n[ALIGN] single-member holdout Macro-F1 (softmax->argmax vs labels at member idx):")
ref = {"k79":0.7617,"k81":0.7567,"dv3ko_s79":0.6851,"dv3ko_s81":0.6899}
for k in ["k79","k81","dv3ko_s79","dv3ko_s81"]:
    p = sm(LOG[k]); pred = p.argmax(1)
    y = np.array([a2id[rlab[real[int(r)]["id"]]] for r in IDX[k]])
    f = f1_score(y,pred,average="macro")
    tag = "MATCH" if (k not in ref or abs(f-ref[k])<0.0006) else "MISMATCH"
    print(f"  {k:11s}: F1={f:.4f}  (reported {ref.get(k,'-')})  [{tag}]")

# idx equality check
print("\n[ALIGN] idx-set / idx-order identity vs k79:")
for k in ["k79","k81","dv3ko_s79","dv3ko_s81"]:
    same_order = np.array_equal(IDX[k], IDX["k79"])
    print(f"  {k:11s} idx == k79 idx (same order)? {same_order}")
print(f"  s48 idx set == hidx set? {set(int(x) for x in IDX['s48'])==set(int(x) for x in IDX['k79'])}")

def member_probs_on(k, inter):
    pos_map={int(r):i for i,r in enumerate(IDX[k])}
    pos=np.array([pos_map[int(r)] for r in inter])
    return sm(LOG[k][pos])

def blend_f1(tf_members, inter):
    seed=np.mean([member_probs_on(k,inter) for k in tf_members],axis=0)
    p_classic=oof[inter]
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
        v=set(int(x) for x in IDX[k]); s=v if s is None else (s & v)
    return np.array(sorted(s))

BASE=["s48","k79","k81"]
combos={"cand_dv79 (k81->dv3ko_s79)":["s48","k79","dv3ko_s79"],
        "cand_dv81 (k81->dv3ko_s81)":["s48","k79","dv3ko_s81"]}

# sanity: v58-core full-blend on full hidx (14000)
inter_full=inter_of(BASE)
bf_plain,bf_se=blend_f1(BASE,inter_full)
print(f"\n[SANITY] v58-core(s48+k79+k81) full-blend on hidx n={len(inter_full)}: plain={bf_plain:.4f} sess-eq={bf_se:.4f} (ref v58-core 0.7702)")

print("\n=== dv3ko k81-SLOT BAND CHECK (deploy blend 0.4*classic + 0.6*mean(3TF softmax)) ===")
print(f"{'combo':30s} {'n_rows':>6s} {'n_sess':>6s} {'base_plain':>10s} {'cand_plain':>10s} {'gate(-.0058)':>12s} {'inband(pl)':>10s} | {'base_se':>7s} {'cand_se':>7s} {'gate_se':>8s} {'inband(se)':>10s}")
for name,tf in combos.items():
    inter=inter_of(tf)
    nsess=len(set(feat.session_of(real[int(r)]["id"]) for r in inter))
    bp,bs=blend_f1(BASE,inter)   # matched v58-core baseline on SAME split
    cp,cs=blend_f1(tf,inter)     # candidate
    gate_p=bp-0.0058; gate_s=bs-0.0058
    ib_p="Y" if cp>=gate_p else "N"; ib_s="Y" if cs>=gate_s else "N"
    print(f"{name:30s} {len(inter):6d} {nsess:6d} {bp:10.4f} {cp:10.4f} {gate_p:12.4f} {ib_p:>10s} | {bs:7.4f} {cs:7.4f} {gate_s:8.4f} {ib_s:>10s}")
