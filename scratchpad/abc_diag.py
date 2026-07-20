import sys, json, numpy as np
sys.path.insert(0, "open/scripts")
import feat

def load(p):
    try: return np.load(p, allow_pickle=False)
    except Exception as e: return f"ERR {e}"

idxfiles = {
 "hidx":"scratchpad/hidx.npy",
 "hidx_pt74c":"scratchpad/hidx_pt74c.npy",
 "hidx_s79":"scratchpad/hidx_s79.npy",
 "hidx_s81":"scratchpad/hidx_s81.npy",
 "h100_va":"scratchpad/h100_va.npy",
 "hv_h83_idx":"scratchpad/hv_h83_idx.npy",
 "classic_v8_holdout_idx":"open/artifacts/classic_v8_holdout_idx.npy",
 "hybrid_holdout_idx_s48":"open/artifacts/hybrid/holdout_idx_s48.npy",
}
idx = {}
print("=== IDX files: shape/dtype ===")
for k,p in idxfiles.items():
    a = load(p); idx[k]=a
    if isinstance(a,str): print(f"{k}: {a}"); continue
    print(f"{k}: shape={a.shape} dtype={a.dtype} min={a.min()} max={a.max()} n_unique={len(np.unique(a))}")

hidx = idx["hidx"]
print("\n=== vs canonical hidx (order-equal? set-equal? |inter|) ===")
for k,a in idx.items():
    if k=="hidx" or isinstance(a,str): continue
    oe = a.shape==hidx.shape and np.array_equal(a,hidx)
    se = np.array_equal(np.sort(a), np.sort(hidx)) if len(a)==len(hidx) else False
    inter = len(np.intersect1d(a,hidx))
    print(f"{k}: order_equal={oe} set_equal={se} |inter_with_hidx|={inter} (len={len(a)})")

# h83 vs h100_va
if not isinstance(idx["hv_h83_idx"],str):
    h83=idx["hv_h83_idx"]; va=idx["h100_va"]
    print(f"\nhv_h83_idx vs h100_va: order_equal={np.array_equal(h83,va)} set_equal={np.array_equal(np.sort(h83),np.sort(va))} |inter|={len(np.intersect1d(h83,va))}")

probfiles = {
 "s48":"open/artifacts/hybrid/holdout_probs_s48.npy",
 "k79":"scratchpad/hp_kf768_79.npy",
 "k81":"scratchpad/hp_kf768_81.npy",
 "k78":"scratchpad/hp_kf768_78.npy",
 "pt74c":"scratchpad/hp_kf768pt74c.npy",
 "h83":"scratchpad/hv_h83_probs.npy",
 "classic_oof":"open/artifacts/oof/oof_classic_probs.npy",
 "classic_v8_holdout":"open/artifacts/classic_v8_holdout_probs.npy",
}
print("\n=== PROB/LOGIT files: shape/dtype/rowsum -> classify ===")
for k,p in probfiles.items():
    a = load(p)
    if isinstance(a,str): print(f"{k}: {a}"); continue
    rs = a.sum(1)
    inrange = float(((a>=0)&(a<=1)).mean())
    cls = "PROBS(sum~1)" if (abs(rs.mean()-1)<0.01 and rs.std()<0.05 and inrange>0.999) else "LOGITS/other"
    print(f"{k}: shape={a.shape} dtype={a.dtype} rowsum[min/mean/max]={rs.min():.3f}/{rs.mean():.3f}/{rs.max():.3f} frac_in[0,1]={inrange:.4f} -> {cls}")

# classic oof ids
ids = json.load(open("open/artifacts/oof/oof_classic_ids.json"))
print(f"\noof_classic_ids: n={len(ids)} sample={ids[:2]}")
cls_v8 = json.load(open("open/artifacts/classic_v8_classes.json"))
print(f"classic_v8_classes: {cls_v8}")
print(f"feat.ACTIONS   : {list(feat.ACTIONS)}")
print(f"ACTIONS==v8classes: {list(feat.ACTIONS)==list(cls_v8)}")
