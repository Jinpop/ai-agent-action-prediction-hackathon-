#!/usr/bin/env python3
"""Compute s48/s51 (and verify-able) holdout probs on hidx rows via local MPS.
Replicates script.py's transformer path exactly (build_transcript + prep.pkl meta + HybridNet head).
Saves scratchpad/hp_s48.npy, hp_s51.npy (softmax probs, ACTIONS order)."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, joblib, torch, torch.nn as nn
sys.path.insert(0, "open/scripts")
import feat
from transformers import AutoModel, AutoTokenizer
import pandas as pd

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEV, flush=True)

def _s(x): return x if isinstance(x, str) else ("" if x is None else str(x))

def build_transcript(sample):  # script.py와 동일
    src = "au" if _s(sample.get("id", "")).startswith("sess_au") else "sim"
    parts = [f"[SRC] {src}"]
    for h in (sample.get("history") or []):
        if not isinstance(h, dict):
            continue
        if h.get("role") == "user":
            parts.append("[U] " + _s(h.get("content", "")))
        elif h.get("role") == "assistant_action":
            args = h.get("args") or {}
            astr = " ".join(f"{k}={_s(v)}" for k, v in args.items())
            parts.append(f"[A] {_s(h.get('name'))} {astr} -> {_s(h.get('result_summary'))}")
    meta = sample.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    parts.append("[META] tier=" + _s(meta.get("user_tier")) + " lang=" + _s(meta.get("language_pref"))
                 + " turn=" + _s(meta.get("turn_index")) + " files=" + ",".join(_s(f) for f in (ws.get("open_files") or []))
                 + " dirty=" + _s(ws.get("git_dirty")) + " ci=" + _s(ws.get("last_ci_status")))
    parts.append("[P] " + _s(sample.get("current_prompt", "")))
    return "\n".join(parts)

class Head(nn.Module):
    def __init__(self, sd):
        super().__init__()
        d_in = sd["net.0.weight"].shape[1] if "net.0.weight" in sd else list(sd.values())[0].shape[1]
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 14))
    def forward(self, x): return self.net(x)

def infer(model_dir, samples, out_path):
    prep = joblib.load(f"{model_dir}/prep.pkl")
    max_len = int(prep.get("max_len", 512))
    tok = AutoTokenizer.from_pretrained(f"{model_dir}/backbone")
    tok.truncation_side = "left"
    net = AutoModel.from_pretrained(f"{model_dir}/backbone").to(DEV).eval()
    sd = torch.load(f"{model_dir}/head.pt", map_location="cpu")
    # head state dict 키 정규화
    ks = list(sd.keys())
    sd2 = {}
    lin = [k for k in ks if k.endswith("weight")]
    sd2["net.0.weight"] = sd[lin[0]]; sd2["net.0.bias"] = sd[lin[0].replace("weight","bias")]
    sd2["net.3.weight"] = sd[lin[1]]; sd2["net.3.bias"] = sd[lin[1].replace("weight","bias")]
    head = Head(sd2); head.net.load_state_dict({k.split("net.")[1]: v for k, v in sd2.items()}, strict=True)
    head = head.to(DEV).eval()
    meta_df = pd.DataFrame([feat.build_meta_row(s) for s in samples])
    M = meta_df.reindex(columns=prep["meta_columns"], fill_value=0.0).values.astype(np.float32)
    M = prep["scaler"].transform(M).astype(np.float32)
    texts = [build_transcript(s) for s in samples]
    B = 24
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), B):
            enc = tok(texts[i:i+B], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(DEV) for k, v in enc.items()}
            cls = net(**enc).last_hidden_state[:, 0]
            m = torch.from_numpy(M[i:i+B]).to(DEV)
            lg = head(torch.cat([cls, m], 1))
            outs.append(torch.softmax(lg.float(), 1).cpu().numpy())
            if (i // B) % 50 == 0:
                print(f"  {model_dir} {i}/{len(texts)}", flush=True)
    P = np.concatenate(outs)
    np.save(out_path, P)
    print(f"saved {out_path} {P.shape}", flush=True)

hidx = np.load("scratchpad/hidx.npy")
real = feat.load_jsonl("open/data/train.jsonl")
sv = [real[i] for i in hidx]
for md, out in [("scratchpad/v45/model_s48", "scratchpad/hp_s48.npy"),
                ("scratchpad/v45/model_s51", "scratchpad/hp_s51.npy")]:
    infer(md, sv, out)
print("DONE", flush=True)
