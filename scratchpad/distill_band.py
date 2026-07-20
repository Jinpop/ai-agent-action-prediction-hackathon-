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
print(f"[CONTRACT] classic OOF: rowsum mean={oof.sum(1).mean():.6f} (already probs, NOT re-softmaxed)")

def sm(z):
    z=z.astype(np.float64); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

# transformer members: (logits_path, idx_path). All raw logits -> per-member softmax.
M = {
 "s48":        ("open/artifacts/hybrid/holdout_probs_s48.npy","open/artifacts/hybrid/holdout_idx_s48.npy"),
 "k79":        ("scratchpad/hp_kf768_79.npy","scratchpad/hidx.npy"),
 "k81":        ("scratchpad/hp_kf768_81.npy","scratchpad/hidx.npy"),
 "mint2b_s79": ("scratchpad/hp_mint2b_s79.npy","scratchpad/hidx.npy"),
 "distill":    ("scratchpad/pack_distill_s48/holdout_probs3.npy","scratchpad/pack_distill_s48/holdout_idx3.npy"),
}
LOG={}; IDX={}
for k,(pp,ip) in M.items():
    LOG[k]=np.load(pp); IDX[k]=np.load(ip)
    assert LOG[k].shape==(14000,14) and IDX[k].shape==(14000,), f"{k} shape {LOG[k].shape} {IDX[k].shape}"

# --- CONTRACT: raw-logits vs probs per member (softmax necessity) ---
print("\n[CONTRACT] per-member raw-logits check (rowsum!=1 => raw logits => softmax applied):")
for k in M:
    rs = LOG[k].sum(1); is_prob = (abs(rs.mean()-1)<1e-3) and (LOG[k].min()>=-1e-6)
    print(f"  {k:11s}: rowsum mean={rs.mean():.4f} min={LOG[k].min():.3f} max={LOG[k].max():.3f} -> {'PROBS(no softmax)' if is_prob else 'RAW LOGITS (softmax applied)'}")

# --- distill holdout_idx3 == hidx check (task-required) ---
hidx = np.load("scratchpad/hidx.npy")
print("\n[SPLIT] distill holdout_idx3 vs local hidx.npy (sha e39aae25; HOLDOUT_IDX=hidx pin):")
same_order = np.array_equal(IDX["distill"], hidx)
same_set = set(int(x) for x in IDX["distill"])==set(int(x) for x in hidx)
print(f"  distill: idx==hidx same_order={same_order} same_set={same_set}")

# --- single-member holdout F1 (uses each member's own idx order) ---
print("\n[ALIGN] single-member holdout Macro-F1 (softmax->argmax vs labels at member idx):")
ref = {"k79":0.7617,"mint2b_s79":0.7650,"distill":0.7521}
for k in ["k79","k81","mint2b_s79","distill"]:
    p = sm(LOG[k]); pred = p.argmax(1)
    y = np.array([a2id[rlab[real[int(r)]["id"]]] for r in IDX[k]])
    f = f1_score(y,pred,average="macro")
    tag = "MATCH" if (k not in ref or abs(f-ref[k])<0.0006) else "MISMATCH"
    print(f"  {k:11s}: F1={f:.6f}  (reported {ref.get(k,'-')})  [{tag}]")

# idx identity vs k79
print("\n[ALIGN] idx-set / idx-order identity vs k79:")
for k in ["k79","k81","mint2b_s79","distill"]:
    print(f"  {k:11s} idx == k79 idx (same order)? {np.array_equal(IDX[k], IDX['k79'])}")
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

# sanity: v58-core on full hidx
v58core=["s48","k79","k81"]; inter_full=inter_of(v58core)
bf_p,bf_s=blend_f1(v58core,inter_full)
print(f"\n[SANITY] v58-core(s48+k79+k81) on hidx n={len(inter_full)}: plain={bf_p:.4f} sess-eq={bf_s:.4f} (abc_band ref 0.7702)")

# ===== CARD1 CHAMPION FRAME band check =====
CHAMP=["s48","k79","mint2b_s79"]                 # current champion (card1)
PROBE=["distill","k79","mint2b_s79"]             # s48 -> distill single swap
inter=inter_of(CHAMP+["distill"])                # common split across all involved
nsess=len(set(feat.session_of(real[int(r)]["id"]) for r in inter))
cp_p,cp_s=blend_f1(CHAMP,inter)   # champion (card1) reproduce -> recon ref 0.7737
pr_p,pr_s=blend_f1(PROBE,inter)   # distill probe
gate_p=cp_p-0.0058; gate_s=cp_s-0.0058
ibp="Y" if pr_p>=gate_p else "N"; ibs="Y" if pr_s>=gate_s else "N"
print("\n=== CARD1 CHAMPION-FRAME BAND CHECK (blend 0.4*classic + 0.6*mean(3TF softmax)) ===")
print(f"  n_rows={len(inter)} n_sess={nsess}")
print(f"  CHAMPION (s48+k79+mint2b_s79)   plain={cp_p:.6f}  sess-eq={cp_s:.6f}   (recon ref 0.7737)")
print(f"  band lower = champ - 0.0058     plain={gate_p:.6f}  sess-eq={gate_s:.6f}")
print(f"  PROBE distill (distill+k79+m2b) plain={pr_p:.6f}  sess-eq={pr_s:.6f}")
print(f"  IN-BAND (plain)={ibp}   IN-BAND (sess-eq)={ibs}")
print(f"  KD delta (probe - champion)     plain={pr_p-cp_p:+.6f}  sess-eq={pr_s-cp_s:+.6f}")

# ===== v58 FRAME reference (plan original: classic + distill + k79 + k81) =====
V58=["s48","k79","k81"]; V58D=["distill","k79","k81"]
inter2=inter_of(V58+["distill"])
v_p,v_s=blend_f1(V58,inter2)
vd_p,vd_s=blend_f1(V58D,inter2)
print("\n=== v58-FRAME REFERENCE (numbers only; plan original s48->distill) ===")
print(f"  n_rows={len(inter2)}")
print(f"  v58 (s48+k79+k81)              plain={v_p:.6f}  sess-eq={v_s:.6f}")
print(f"  v58-distill (distill+k79+k81)  plain={vd_p:.6f}  sess-eq={vd_s:.6f}")
print(f"  KD delta (v58-distill - v58)   plain={vd_p-v_p:+.6f}  sess-eq={vd_s-v_s:+.6f}")
