"""Kaggle/Colab GPU 용 v14 — klue/roberta-large 하이브리드 (풀 트랜스크립트).

v12(base) 와 동일한 입력(풀 트랜스크립트 + 메타 concat + left-truncation)에
백본만 roberta-base(110M) -> roberta-large(337M) 로 교체.
  * BATCH 16 x GRAD_ACCUM 4 = 실효 배치 64 (T4 16GB 용)
  * LR 1e-5 (large 는 낮은 LR 필수 — 2e-5 면 발산 위험)
  * 제출 zip 은 fp16 저장 (fp32 1.35GB > 1GB 제한 -> fp16 ~680MB)
  * OOM 시: GRAD_CKPT=1 (gradient checkpointing, ~30% 느려짐) or BATCH=8 GRAD_ACCUM=8

[ Kaggle 실행법 (Accelerator: GPU T4 x2) ]
  1) 데이터셋(dacon236694) 연결 후:
     BASE = "/kaggle/input/datasets/jinpop/dacon236694"
     !cp {BASE}/feat.py {BASE}/colab_train_large.py .
     !mkdir -p data && cp {BASE}/colab_data/* data/ && wc -l data/train.jsonl
  2) !pip -q install "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece scikit-learn joblib
  3) !CUDA_VISIBLE_DEVICES=0 python colab_train_large.py     # 단일 GPU 강제(중요! DP 오버헤드 방지)
     (T4 단일 기준 총 6~8시간. 시간 없으면 EPOCHS=2)
  4) 산출물: submit_large.zip / holdout_probs_L.npy / holdout_idx_L.npy

주의: 로컬에서 이 파일을 import 하지 말 것 (import 즉시 학습 시작됨).
"""
import json
import os
import shutil
import zipfile

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
from transformers import (AutoModel, AutoTokenizer, Trainer, TrainingArguments)
from transformers.modeling_outputs import SequenceClassifierOutput

import feat  # 같이 업로드 필수

# ================== 설정 ==================
MODEL_NAME = os.environ.get("MODEL_NAME", "klue/roberta-large")
DATA_DIR = "./data"
SUB_DIR = "./model_sub"
MAX_LEN = int(os.environ.get("MAX_LEN", 384))
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH = int(os.environ.get("BATCH", 16))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 4))
LR = float(os.environ.get("LR", 1e-5))
SEED = 42
N_LIMIT = int(os.environ.get("N_LIMIT", 0))
SKIP_REFIT = os.environ.get("SKIP_REFIT", "0") == "1"
GRAD_CKPT = os.environ.get("GRAD_CKPT", "0") == "1"

ACTIONS = feat.ACTIONS
lab2id = {a: i for i, a in enumerate(ACTIONS)}


def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def build_transcript(sample):
    """v12 와 동일한 직렬화 (스태킹 시 재현 일치 필수 — 수정 금지)."""
    parts = []
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
    parts.append(
        f"[META] tier={_s(meta.get('user_tier'))} ci={_s(ws.get('last_ci_status'))} "
        f"dirty={int(bool(ws.get('git_dirty')))} turn={meta.get('turn_index', 0)} "
        f"open={','.join(_s(p) for p in (ws.get('open_files') or [])[:6])}")
    parts.append("[P] " + _s(sample.get("current_prompt", "")))
    return "\n".join(parts)


# ---------- 데이터 로드 ----------
print(f"Model: {MODEL_NAME}  (v14 large hybrid, MAX_LEN={MAX_LEN}, "
      f"batch {BATCH}x{GRAD_ACCUM}, lr={LR})")
train = feat.load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
labels_df = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
lab = dict(zip(labels_df["id"], labels_df["action"]))
if N_LIMIT:
    train = train[:N_LIMIT]
    print(f"[smoke] subset {N_LIMIT}")

ids = [s["id"] for s in train]
texts = [build_transcript(s) for s in train]
y = np.array([lab2id[lab[i]] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])

print("Build meta features...")
meta_df = feat.build_meta_frame(train)
META_COLS = list(meta_df.columns)
M = meta_df.values.astype(np.float32)
print(f"  meta dims = {M.shape[1]}")

tr, va = next(GroupKFold(n_splits=5).split(texts, y, groups))
print(f"train={len(tr)}  holdout={len(va)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.truncation_side = "left"   # [META]/[P] 보존, 오래된 history 부터 자름


def make_ds(idx, scaler):
    Ms = scaler.transform(M[idx]).astype(np.float32)
    d = Dataset.from_dict({
        "text": [texts[i] for i in idx],
        "label": [int(y[i]) for i in idx],
        "meta": [row.tolist() for row in Ms],
    })
    return d.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                 batched=True, remove_columns=["text"])


def sqrt_weights(yy):
    counts = np.bincount(yy, minlength=len(ACTIONS)).astype(np.float64)
    w = np.sqrt(counts.sum() / (len(ACTIONS) * np.maximum(counts, 1)))
    return torch.tensor((w / w.mean()).astype(np.float32))


class HybridNet(nn.Module):
    def __init__(self, name, n_meta, n_labels, class_weights=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        h = self.backbone.config.hidden_size          # large = 1024
        self.head = nn.Sequential(
            nn.Linear(h + n_meta, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_labels))
        self.register_buffer("cw", class_weights if class_weights is not None
                             else torch.ones(n_labels))

    def forward(self, input_ids=None, attention_mask=None, meta=None, labels=None):
        cls = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.head(torch.cat([cls.float(), meta.float()], dim=-1))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.cw)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}


def train_model(idx_tr, idx_va=None):
    scaler = StandardScaler().fit(M[idx_tr])
    ds_tr = make_ds(idx_tr, scaler)
    ds_va = make_ds(idx_va, scaler) if idx_va is not None else None
    model = HybridNet(MODEL_NAME, M.shape[1], len(ACTIONS), sqrt_weights(y[idx_tr]))
    if GRAD_CKPT:
        model.backbone.gradient_checkpointing_enable()
    args = TrainingArguments(
        output_dir="./_ckpt", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=32,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch" if ds_va is not None else "no",
        save_strategy="no", logging_steps=50,
        fp16=torch.cuda.is_available(), seed=SEED, report_to="none",
        remove_unused_columns=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr,
                      eval_dataset=ds_va, compute_metrics=metrics if ds_va is not None else None,
                      processing_class=tok)
    trainer.train()
    return trainer, model, scaler


# ---------- 홀드아웃 학습/평가 ----------
trainer, model, scaler = train_model(tr, va)
out = trainer.predict(make_ds(va, scaler))
pred = out.predictions.argmax(-1)
f1 = f1_score(y[va], pred, average="macro")
print(f"\n==> 홀드아웃 Macro-F1 = {f1:.4f}   (v12 base 트랜스크립트와 비교)")
print(classification_report(y[va], pred, target_names=ACTIONS, digits=3, zero_division=0))
np.save("holdout_probs_L.npy", out.predictions)
np.save("holdout_idx_L.npy", va)

# ---------- 전체 재학습(제출용) ----------
if SKIP_REFIT:
    print("[smoke] SKIP_REFIT=1 -> 홀드아웃 모델 저장")
    model_full, scaler_full = model, scaler
else:
    print("\nRefit on FULL data...")
    del trainer, model
    torch.cuda.empty_cache()
    _, model_full, scaler_full = train_model(np.arange(len(texts)))

if os.path.exists(SUB_DIR):
    shutil.rmtree(SUB_DIR)
os.makedirs(SUB_DIR)
model_full.backbone.half().save_pretrained(os.path.join(SUB_DIR, "backbone"))  # fp16 저장 (1GB 제한)
tok.save_pretrained(os.path.join(SUB_DIR, "backbone"))
_tc = os.path.join(SUB_DIR, "backbone", "tokenizer_config.json")
if os.path.exists(_tc):   # transformers 5.x 신형 포맷 -> 구버전 호환 패치 (v12 제출오류 재발 방지)
    _cfg = json.load(open(_tc))
    _cfg.pop("backend", None)
    _cfg.pop("is_local", None)
    _cfg["tokenizer_class"] = "BertTokenizerFast"
    json.dump(_cfg, open(_tc, "w"), ensure_ascii=False, indent=2)
torch.save({k: v.float() for k, v in model_full.head.state_dict().items()},
           os.path.join(SUB_DIR, "head.pt"))                                  # head 는 fp32 유지
joblib.dump({"meta_columns": META_COLS, "scaler": scaler_full,
             "actions": ACTIONS, "max_len": MAX_LEN},
            os.path.join(SUB_DIR, "prep.pkl"))
print(f"saved -> {SUB_DIR}")

# ---------- 제출용 script.py ----------
SUBMIT_SCRIPT = r'''"""추론(제출용) — v14 large 하이브리드(풀 트랜스크립트 + meta, fp16 백본)."""
import csv, os
import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import feat

DEV = "cuda" if torch.cuda.is_available() else "cpu"
prep = joblib.load("./model_sub/prep.pkl")
ACTIONS, MAX_LEN = prep["actions"], prep["max_len"]


def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def build_transcript(sample):
    parts = []
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
    parts.append(
        f"[META] tier={_s(meta.get('user_tier'))} ci={_s(ws.get('last_ci_status'))} "
        f"dirty={int(bool(ws.get('git_dirty')))} turn={meta.get('turn_index', 0)} "
        f"open={','.join(_s(p) for p in (ws.get('open_files') or [])[:6])}")
    parts.append("[P] " + _s(sample.get("current_prompt", "")))
    return "\n".join(parts)


class HybridNet(nn.Module):
    def __init__(self, path, n_meta, n_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)   # fp16 로 저장됨
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h + n_meta, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_labels))


tok = AutoTokenizer.from_pretrained("./model_sub/backbone")
tok.truncation_side = "left"
samples = feat.load_jsonl("./data/test.jsonl")
texts = [build_transcript(s) for s in samples]
Mt = prep["scaler"].transform(
    feat.build_meta_frame(samples, columns=prep["meta_columns"]).values.astype(np.float32)
).astype(np.float32)

net = HybridNet("./model_sub/backbone", Mt.shape[1], len(ACTIONS))
net.head.load_state_dict(torch.load("./model_sub/head.pt", map_location="cpu"))
if DEV == "cpu":
    net.backbone = net.backbone.float()   # CPU 는 fp16 연산 불가 대비
net = net.to(DEV).eval()

preds = []
with torch.no_grad():
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i + 32], truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt").to(DEV)
        m = torch.tensor(Mt[i:i + 32]).to(DEV)
        cls = net.backbone(**b).last_hidden_state[:, 0].float()
        preds.extend(net.head(torch.cat([cls, m], -1)).argmax(-1).cpu().tolist())
pm = {s.get("id", ""): ACTIONS[p] for s, p in zip(samples, preds)}

with open("./data/sample_submission.csv", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    fields, rows = r.fieldnames, list(r)
for row in rows:
    if row["id"] in pm:
        row["action"] = pm[row["id"]]
os.makedirs("./output", exist_ok=True)
with open("./output/submission.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("saved ./output/submission.csv", len(rows))
'''
open("script.py", "w", encoding="utf-8").write(SUBMIT_SCRIPT)
open("requirements.txt", "w", encoding="utf-8").write(
    "transformers==4.57.6\ntorch\nsentencepiece\nscikit-learn==1.6.1\n"
    "joblib==1.5.3\npandas==2.3.3\nnumpy==2.0.2\n")

zp = "submit_large.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("script.py")
    z.write("feat.py")
    z.write("requirements.txt")
    for root, _, files in os.walk(SUB_DIR):
        for fn in files:
            fp = os.path.join(root, fn)
            z.write(fp, os.path.relpath(fp, "."))
size = os.path.getsize(zp) / 1e6
print(f"\n[완료] {zp}  ({size:.0f} MB)  홀드아웃 Macro-F1={f1:.4f}")
if size > 950:
    print("경고: 1GB 근접! 확인 필요.")
