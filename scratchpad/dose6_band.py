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
 "mint2b_s79": ("scratchpad/hp_mint2b_s79.npy","scratchpad/hidx.npy"),
 "dose6":      ("scratchpad/dose6_fetch/holdout_probs3.npy","scratchpad/dose6_fetch/holdout_idx3.npy"),
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

# --- dose6 holdout_idx3 == hidx check (task-required) ---
hidx = np.load("scratchpad/hidx.npy")
same_order = np.array_equal(IDX["dose6"], hidx)
same_set = set(int(x) for x in IDX["dose6"])==set(int(x) for x in hidx)
print(f"\n[SPLIT] dose6 idx3 == hidx  same_order={same_order}  same_set={same_set}")

# --- single-member holdout F1 (each member's own idx order) ---
print("\n[SINGLE-MEMBER] holdout Macro-F1 (softmax->argmax vs labels at member idx):")
ref = {"k79":0.7617,"mint2b_s79":0.7650,"dose6":0.7590}
res={}
for k in ["k79","mint2b_s79","dose6"]:
    p = sm(LOG[k]); pred = p.argmax(1)
    y = np.array([a2id[rlab[real[int(r)]["id"]]] for r in IDX[k]])
    f = f1_score(y,pred,average="macro"); res[k]=f
    tag = "MATCH" if (k not in ref or abs(f-ref[k])<0.0006) else "CHECK"
    print(f"  {k:11s}: F1={f:.6f}  (reported {ref.get(k,'-')})  [{tag}]")
# also compute dose6 F1 WITHOUT softmax to confirm argmax-invariance
p_raw=LOG["dose6"]; pred_raw=p_raw.argmax(1)
y6=np.array([a2id[rlab[real[int(r)]["id"]]] for r in IDX["dose6"]])
print(f"  dose6 raw-logit argmax F1 = {f1_score(y6,pred_raw,average='macro'):.6f} (argmax invariance check)")
print(f"  SINGLE delta dose6 - mint2b_s79 = {res['dose6']-res['mint2b_s79']:+.6f} (ref -0.0060)")

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

# ===== CARD1 CHAMPION FRAME band check (mint2b_s79 slot -> dose6 single swap) =====
CHAMP=["s48","k79","mint2b_s79"]                 # current champion (card1 = v58 with k81->mint2b_s79)
PROBE=["s48","k79","dose6"]                       # mint2b_s79 -> dose6 single swap
inter=inter_of(CHAMP+["dose6"])                   # common split across all involved
nsess=len(set(feat.session_of(real[int(r)]["id"]) for r in inter))
cp_p,cp_s=blend_f1(CHAMP,inter)   # champion reproduce -> recon ref 0.773710
pr_p,pr_s=blend_f1(PROBE,inter)   # dose6 probe

# Band anchor: task-specified fixed recon champion 0.773710, lower 0.767910
RECON_CHAMP=0.773710; RECON_LOWER=0.767910
gate_p_repro=cp_p-0.0058; gate_s_repro=cp_s-0.0058
ibp_recon="Y" if pr_p>=RECON_LOWER else "N"
ibp_repro="Y" if pr_p>=gate_p_repro else "N"
ibs_repro="Y" if pr_s>=gate_s_repro else "N"
print("\n=== CARD1 CHAMPION-FRAME BAND CHECK (blend 0.4*classic + 0.6*mean(3TF softmax)) ===")
print(f"  members: CHAMP={CHAMP}  PROBE={PROBE}")
print(f"  n_rows={len(inter)} n_sess={nsess}")
print(f"  CHAMPION (s48+k79+mint2b_s79) reproduce  plain={cp_p:.6f}  sess-eq={cp_s:.6f}   (recon ref 0.773710)")
print(f"  PROBE dose6 (s48+k79+dose6)              plain={pr_p:.6f}  sess-eq={pr_s:.6f}")
print(f"  --- band vs FIXED recon anchor (task): lower = 0.773710 - 0.0058 = {RECON_LOWER:.6f} ---")
print(f"      IN-BAND (plain, recon anchor) = {ibp_recon}")
print(f"  --- band vs REPRODUCED champion on this split ---")
print(f"      lower plain={gate_p_repro:.6f} sess-eq={gate_s_repro:.6f}")
print(f"      IN-BAND (plain)={ibp_repro}   IN-BAND (sess-eq)={ibs_repro}")
print(f"  dose delta (probe - champion)  plain={pr_p-cp_p:+.6f}  sess-eq={pr_s-cp_s:+.6f}")
print(f"  probe margin over recon lower  plain={pr_p-RECON_LOWER:+.6f}")
