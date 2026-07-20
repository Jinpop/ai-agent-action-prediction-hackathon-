#!/usr/bin/env python3
"""Refit classic (HGB/LogReg/NB) with each mint metadata variant, eval classic-blend holdout.

Replicates v39 champion recipe exactly (params pulled from v39 artifacts.pkl):
  vec0 word(1,2)/50k/min_df4/sublinear, vec1 char_wb(3,5)/40k/min_df4/sublinear,
  svd 200, HGB lr0.1/500/63/l2=1, LogReg C3 balanced, ComplementNB a0.3, MaxAbsScaler.
  meta = build_meta_frame (119) + 9 relational = 128. blend 0.45/0.40/0.15.
Split: GroupKFold(5) first fold on real train (mint added to train, holdout sessions excluded).

Text (and thus vec/svd/NB) is IDENTICAL across variants (only metadata differs) -> fit once, reuse.
Only meta-dependent HGB/LogReg refit per variant.
"""
import sys, csv, json
import numpy as np
sys.path.insert(0, "open/scripts")
import feat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from scipy.sparse import hstack as sp_hstack, csr_matrix
import re as _re

REAL = "open/data/train.jsonl"
REAL_LAB = "open/data/train_labels.csv"
MINT_LAB = "open/data/train_mint_labels.csv"
VARIANTS = {
    "baseline_mint1": "open/data/train_mint.jsonl",
    "ci_only":        "open/data/train_mint_ci.jsonl",
    "full_recon":     "open/data/train_mint_full.jsonl",
}

# ---- relational feats (identical to v39 script.py / refit_rel.py) ----
_FILE_RE = _re.compile(r'[\w./-]+\.\w{1,5}'); _GLOB_RE = _re.compile(r'[*?]|\*\.\w+|\*\*')
REL_COLS = ["rel_glob","rel_nfile","rel_wasread","rel_pastpath","rel_lastsib","rel_lastfail","rel_nopen","rel_dirword","rel_searchword"]
def _sr(x): return x if isinstance(x, str) else ("" if x is None else str(x))
def _relfeats(s):
    cp=_sr(s.get("current_prompt","")); cpl=cp.lower(); h=s.get("history") or []
    w=(s.get("session_meta") or {}).get("workspace") or {}; of=[_sr(f) for f in (w.get("open_files") or [])]
    past_reads=set(); past_paths=set(); last_result=""; last_action=None
    for e in h:
        if not isinstance(e,dict) or e.get("role")!="assistant_action": continue
        nm=e.get("name"); a=e.get("args") or {}; last_action=nm
        pth=_sr(a.get("path") or a.get("target") or "")
        if pth: past_paths.add(pth.lower())
        if nm=="read_file" and pth: past_reads.add(pth.lower())
        last_result=_sr(e.get("result_summary",""))
    return [
        1.0 if _GLOB_RE.search(cp) else 0.0, float(len(_FILE_RE.findall(cp))),
        1.0 if any(_sr(x).split("/")[-1].split(".")[0].lower() in cpl for x in past_reads) else 0.0,
        1.0 if any(_sr(x).split("/")[-1].split(".")[0].lower() in cpl for x in past_paths) else 0.0,
        1.0 if last_action in ("read_file","grep_search","glob_pattern","list_directory") else 0.0,
        1.0 if ("error" in last_result.lower() or "fail" in last_result.lower()) else 0.0,
        float(len(of)),
        1.0 if any(k in cpl for k in ["폴더","디렉","directory","folder","목록","list","ls "]) else 0.0,
        1.0 if any(k in cpl for k in ["찾","검색","어디","where","grep","search","usage","참조","정의"]) else 0.0,
    ]

def load_labels(path):
    d = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r: d[row["id"]] = row["action"]
    return d

def src_text(s):
    return ("[SRC] " + ("au" if str(s.get("id","")).startswith("sess_au") else "sim") + " ") + feat.build_text(s)

def build_meta(samples, columns):
    base = feat.build_meta_frame(samples)
    rel = np.array([_relfeats(s) for s in samples], dtype=np.float32)
    for j, c in enumerate(REL_COLS): base[c] = rel[:, j]
    return base.reindex(columns=columns, fill_value=0.0).values.astype(np.float32)

def main():
    ACT = feat.ACTIONS
    a2id = {a: i for i, a in enumerate(ACT)}
    real = feat.load_jsonl(REAL)
    real_lab = load_labels(REAL_LAB)
    mint_lab = load_labels(MINT_LAB)
    N_REAL = len(real)
    print(f"real={N_REAL}")

    # meta columns = v39's exact 128
    import joblib
    META_COLS = list(joblib.load("scratchpad/v39x/model/artifacts.pkl")["meta_columns"])

    real_ids = [s["id"] for s in real]
    y_real = np.array([a2id[real_lab[i]] for i in real_ids])
    groups_real = np.array([feat.session_of(i) for i in real_ids])
    tr0, va = next(GroupKFold(n_splits=5).split(real_ids, y_real, groups_real))
    va_sess = set(groups_real[va])
    print(f"holdout={len(va)} sessions={len(va_sess)}")

    # text is identical across variants -> load mint texts once (from baseline; texts same)
    mint = feat.load_jsonl(VARIANTS["baseline_mint1"])
    mint_ids = [s["id"] for s in mint]
    y_mint = np.array([a2id[mint_lab[i]] for i in mint_ids])
    groups_mint = np.array([feat.session_of(i) for i in mint_ids])
    mint_keep = np.array([k for k in range(len(mint)) if groups_mint[k] not in va_sess])
    print(f"mint kept in train: {len(mint_keep)}/{len(mint)}")

    # --- shared text features (identical across variants) ---
    all_samples = real + mint
    texts = [src_text(s) for s in all_samples]
    tr = np.concatenate([tr0, (N_REAL + mint_keep).astype(tr0.dtype)])
    y_all = np.concatenate([y_real, y_mint])

    print("fit vectorizers...")
    vecs = [
        TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=50000, min_df=4, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=40000, min_df=4, sublinear_tf=True),
    ]
    Xtr_txt = sp_hstack([v.fit_transform([texts[i] for i in tr]) for v in vecs]).tocsr()
    Xva_txt = sp_hstack([v.transform([texts[i] for i in va]) for v in vecs]).tocsr()
    print("fit svd..."); svd = TruncatedSVD(n_components=200, random_state=0)
    Xtr_svd = svd.fit_transform(Xtr_txt).astype(np.float32)
    Xva_svd = svd.transform(Xva_txt).astype(np.float32)
    print("fit NB..."); nb = ComplementNB(alpha=0.3); nb.fit(Xtr_txt, y_all[tr])
    p_nb = nb.predict_proba(Xva_txt)
    y_va = y_real[va]

    # HGB sample weights: sqrt inverse class freq (recipe: sqrt class weights)
    def sqrt_w(yy):
        cnt = np.bincount(yy, minlength=len(ACT)).astype(np.float64); cnt[cnt==0]=1
        w_c = np.sqrt(cnt.sum()/cnt); w_c/=w_c.mean()
        return w_c[yy]
    sw = sqrt_w(y_all[tr])

    results = {}
    for name, path in VARIANTS.items():
        samples_v = real + feat.load_jsonl(path)
        Mtr = build_meta(samples_v, META_COLS)  # build for all, then index
        Xtr_meta = Mtr[tr]; Xva_meta = Mtr[va]
        scaler = MaxAbsScaler().fit(Xtr_meta)
        # HGB: [svd, meta]
        hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=500, max_leaf_nodes=63,
                                             l2_regularization=1.0, random_state=0)
        hgb.fit(np.hstack([Xtr_svd, Xtr_meta]), y_all[tr], sample_weight=sw)
        p_hgb = hgb.predict_proba(np.hstack([Xva_svd, Xva_meta]))
        # LogReg: [text, scaled_meta]
        Xtr_lr = sp_hstack([Xtr_txt, csr_matrix(scaler.transform(Xtr_meta))]).tocsr()
        Xva_lr = sp_hstack([Xva_txt, csr_matrix(scaler.transform(Xva_meta))]).tocsr()
        lr = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced")
        lr.fit(Xtr_lr, y_all[tr])
        p_lr = lr.predict_proba(Xva_lr)
        # align class order (models trained on y in ACT ids already) -> classes_ = sorted ids
        def align(clf, p):
            return p[:, [list(clf.classes_).index(k) for k in range(len(ACT))]]
        blend = 0.45*align(hgb,p_hgb) + 0.40*align(lr,p_lr) + 0.15*align(nb,p_nb)
        f1 = f1_score(y_va, blend.argmax(1), average="macro")
        results[name] = f1
        print(f"  [{name}] classic-blend holdout Macro-F1 = {f1:.4f}")

    print("\n=== SUMMARY ===")
    base = results["baseline_mint1"]
    for name, f1 in results.items():
        print(f"  {name:16s} {f1:.4f}  ({f1-base:+.4f} vs baseline)")

if __name__ == "__main__":
    main()
