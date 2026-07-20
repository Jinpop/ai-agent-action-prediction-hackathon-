#!/usr/bin/env python3
"""H100 va(sk1.7.2 fold0, 14000행)에서 s48/k79/k78/k81 홀드아웃 확률 사전추론 + 베이스라인.
21:30 H100 밴드체크를 즉시화. mps_infer.py의 build_transcript/Head/infer와 동일 경로."""
import sys, csv, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np, joblib, torch, torch.nn as nn, pandas as pd
sys.path.insert(0, "open/scripts")
import feat
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
def _s(x): return x if isinstance(x, str) else ("" if x is None else str(x))
def build_transcript(sample):
    src = "au" if _s(sample.get("id","")).startswith("sess_au") else "sim"
    parts = [f"[SRC] {src}"]
    for h in (sample.get("history") or []):
        if not isinstance(h, dict): continue
        if h.get("role") == "user": parts.append("[U] " + _s(h.get("content","")))
        elif h.get("role") == "assistant_action":
            args = h.get("args") or {}
            astr = " ".join(f"{k}={_s(v)}" for k,v in args.items())
            parts.append(f"[A] {_s(h.get('name'))} {astr} -> {_s(h.get('result_summary'))}")
    meta = sample.get("session_meta") or {}; ws = meta.get("workspace") or {}
    parts.append("[META] tier=" + _s(meta.get("user_tier")) + " lang=" + _s(meta.get("language_pref"))
                 + " turn=" + _s(meta.get("turn_index")) + " files=" + ",".join(_s(f) for f in (ws.get("open_files") or []))
                 + " dirty=" + _s(ws.get("git_dirty")) + " ci=" + _s(ws.get("last_ci_status")))
    parts.append("[P] " + _s(sample.get("current_prompt","")))
    return "\n".join(parts)

class Head(nn.Module):
    def __init__(self, d_in): super().__init__(); self.net = nn.Sequential(nn.Linear(d_in,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,14))
    def forward(self,x): return self.net(x)

def infer(model_dir, samples):
    prep = joblib.load(f"{model_dir}/prep.pkl"); max_len = int(prep.get("max_len",512))
    tok = AutoTokenizer.from_pretrained(f"{model_dir}/backbone"); tok.truncation_side = "left"
    net = AutoModel.from_pretrained(f"{model_dir}/backbone").to(DEV).eval()
    sd = torch.load(f"{model_dir}/head.pt", map_location="cpu")
    lin = [k for k in sd if k.endswith("weight")]
    d_in = sd[lin[0]].shape[1]; head = Head(d_in)
    head.net.load_state_dict({"0.weight":sd[lin[0]],"0.bias":sd[lin[0].replace('weight','bias')],
                              "3.weight":sd[lin[1]],"3.bias":sd[lin[1].replace('weight','bias')]}, strict=True)
    head = head.to(DEV).eval()
    meta_df = pd.DataFrame([feat.build_meta_row(s) for s in samples])
    M = meta_df.reindex(columns=prep["meta_columns"], fill_value=0.0).values.astype(np.float32)
    M = prep["scaler"].transform(M).astype(np.float32)
    texts = [build_transcript(s) for s in samples]; B = 24; outs = []
    with torch.no_grad():
        for i in range(0, len(texts), B):
            enc = tok(texts[i:i+B], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k:v.to(DEV) for k,v in enc.items()}
            cls = net(**enc).last_hidden_state[:,0]
            m = torch.from_numpy(M[i:i+B]).to(DEV)
            lg = head(torch.cat([cls,m],1))
            outs.append(torch.softmax(lg.float(),1).cpu().numpy())
            if (i//B)%80==0: print(f"  {model_dir} {i}/{len(texts)}", flush=True)
    return np.concatenate(outs)

va = np.load("scratchpad/h100_va.npy")
real = feat.load_jsonl("open/data/train.jsonl")
rlab = {r["id"]: r["action"] for r in csv.DictReader(open("open/data/train_labels.csv"))}
ACT = feat.ACTIONS; a2id = {a:i for i,a in enumerate(ACT)}
sv = [real[i] for i in va]
y = np.array([a2id[rlab[real[i]["id"]]] for i in va])
g = np.array([feat.session_of(real[i]["id"]) for i in va])
cnt = collections.Counter(g); w = np.array([1.0/cnt[x] for x in g])
np.save("scratchpad/hva_y.npy", y); np.save("scratchpad/hva_w.npy", w)

models = {"s48":"scratchpad/v45/model_s48","k79":"scratchpad/v49/model_kf768_79",
          "k78":"scratchpad/v49/model_kf768_78","k81":"scratchpad/stage_v58/model_kf768_81"}
P = {}
for name, md in models.items():
    P[name] = infer(md, sv); np.save(f"scratchpad/hva_{name}.npy", P[name])
    print(f"[done] {name} {P[name].shape}", flush=True)

def se(core): return f1_score(y, core.argmax(1), average="macro", sample_weight=w)
b_v49 = se((P["s48"]+P["k79"]+P["k78"])/3)
b_v58 = se((P["s48"]+P["k79"]+P["k81"])/3)
print(f"\n[baseline] v49-core s48+k79+k78 sess-eq={b_v49:.4f} (gate={b_v49-0.0058:.4f})")
print(f"[baseline] v58-core s48+k79+k81 sess-eq={b_v58:.4f} (gate={b_v58-0.0058:.4f})")
print("BAND_PREP_DONE", flush=True)
