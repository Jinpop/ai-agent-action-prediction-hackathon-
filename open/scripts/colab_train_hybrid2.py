"""Colab GPU 용 하이브리드 v2 — 풀 트랜스크립트 직렬화 (Dacon 236694).

v10(홀드아웃 0.6519 / LB 0.6334) 대비 핵심 변경:
  * history 를 행동 이름 토큰(seq_*)으로 뭉개지 않고, user발화<->행동을
    교차 순서 그대로 + args + result_summary 까지 대화록 형태로 직렬화.
    ("[U] 요청... [A] run_tests target=all -> PASS: 53 tests" 식)
    -> "테스트 PASS 후엔 X, FAIL 후엔 Y" 같은 인과 신호를 모델이 직접 봄.
  * MAX_LEN 256 -> 384 (트랜스크립트 수용), EPOCHS 기본 4
  * CLS+메타 concat head, sqrt 클래스 가중치는 v10 그대로 유지.

[ 실행법 (Google Colab, 런타임 > GPU(T4) 선택) ]
  1) colab_data.zip + feat.py + 이 파일 업로드 후:
        !rm -rf data && mkdir data && unzip -o colab_data.zip -d data/
        !wc -l data/train.jsonl   # 70000 확인
  2) !pip -q install "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece scikit-learn joblib
  3) !python colab_train_hybrid2.py     (T4 기준 총 3~4시간)
  4) 끝나면 submit_hybrid2.zip / holdout_probs2.npy / holdout_idx2.npy 다운로드

산출물: ./model_sub/, ./submit_hybrid2.zip, holdout_probs2.npy, holdout_idx2.npy
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

import feat  # 고전 ML 과 동일한 피처 모듈 (같이 업로드 필수)

# ================== 설정 ==================
MODEL_NAME = os.environ.get("MODEL_NAME", "klue/roberta-base")
DATA_DIR = "./data"
SUB_DIR = "./model_sub"
MAX_LEN = int(os.environ.get("MAX_LEN", 384))
EPOCHS = int(os.environ.get("EPOCHS", 4))
BATCH = int(os.environ.get("BATCH", 32))
LR = float(os.environ.get("LR", 2e-5))
SEED = 42
N_LIMIT = int(os.environ.get("N_LIMIT", 0))      # >0 이면 subset (smoke test)
SKIP_REFIT = os.environ.get("SKIP_REFIT", "0") == "1"

ACTIONS = feat.ACTIONS
lab2id = {a: i for i, a in enumerate(ACTIONS)}


def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def build_transcript(sample):
    """세션을 대화록 그대로 직렬화: user발화/행동(args,결과) 교차 + 메타 + 현재 프롬프트."""
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
print(f"Model: {MODEL_NAME}  (hybrid v2: full transcript + meta, MAX_LEN={MAX_LEN})")
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
# 트랜스크립트는 43%가 384토큰 초과. 기본(오른쪽 자름)이면 맨 뒤의
# [P] 현재 프롬프트가 39% 샘플에서 통째로 잘림(v12 1차 0.5459 붕괴 원인).
# -> 왼쪽(오래된 history)부터 자르고 [META]/[P]는 항상 보존.
tok.truncation_side = "left"


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
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h + n_meta, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_labels))
        self.register_buffer("cw", class_weights if class_weights is not None
                             else torch.ones(n_labels))

    def forward(self, input_ids=None, attention_mask=None, meta=None, labels=None):
        cls = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.head(torch.cat([cls, meta.float()], dim=-1))
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
    args = TrainingArguments(
        output_dir="./_ckpt", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        learning_rate=LR, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch" if ds_va is not None else "no",
        save_strategy="no", logging_steps=100,
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
print(f"\n==> 홀드아웃 Macro-F1 = {f1:.4f}   (v10 하이브리드 0.6519 / v7 고전ML 0.6089)")
print(classification_report(y[va], pred, target_names=ACTIONS, digits=3, zero_division=0))
np.save("holdout_probs2.npy", out.predictions)   # 스태킹용 (로짓)
np.save("holdout_idx2.npy", va)

# ---------- 전체 재학습(제출용) ----------
if SKIP_REFIT:
    print("[smoke] SKIP_REFIT=1 -> 홀드아웃 모델 저장")
    model_full, scaler_full = model, scaler
else:
    print("\nRefit on FULL data...")
    _, model_full, scaler_full = train_model(np.arange(len(texts)))

if os.path.exists(SUB_DIR):
    shutil.rmtree(SUB_DIR)
os.makedirs(SUB_DIR)
model_full.backbone.save_pretrained(os.path.join(SUB_DIR, "backbone"))
tok.save_pretrained(os.path.join(SUB_DIR, "backbone"))
# transformers 5.x 는 tokenizer_class 를 "TokenizersBackend"(신형)로 저장하는데
# 구버전 transformers 가 못 읽음(v12 1차 제출 오류) -> 클래식 클래스명으로 패치
_tc = os.path.join(SUB_DIR, "backbone", "tokenizer_config.json")
if os.path.exists(_tc):
    _cfg = json.load(open(_tc))
    _cfg.pop("backend", None)
    _cfg.pop("is_local", None)
    _cfg["tokenizer_class"] = "BertTokenizerFast"
    json.dump(_cfg, open(_tc, "w"), ensure_ascii=False, indent=2)
torch.save(model_full.head.state_dict(), os.path.join(SUB_DIR, "head.pt"))
joblib.dump({"meta_columns": META_COLS, "scaler": scaler_full,
             "actions": ACTIONS, "max_len": MAX_LEN},
            os.path.join(SUB_DIR, "prep.pkl"))
print(f"saved -> {SUB_DIR}")

# ---------- 제출용 script.py ----------
SUBMIT_SCRIPT = r'''"""추론(제출용) — 하이브리드 v2(풀 트랜스크립트 + meta)로 test.jsonl 예측."""
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
        self.backbone = AutoModel.from_pretrained(path)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h + n_meta, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_labels))


tok = AutoTokenizer.from_pretrained("./model_sub/backbone")
tok.truncation_side = "left"   # 학습과 동일: 오래된 history 부터 자름
samples = feat.load_jsonl("./data/test.jsonl")
texts = [build_transcript(s) for s in samples]
Mt = prep["scaler"].transform(
    feat.build_meta_frame(samples, columns=prep["meta_columns"]).values.astype(np.float32)
).astype(np.float32)

net = HybridNet("./model_sub/backbone", Mt.shape[1], len(ACTIONS))
net.head.load_state_dict(torch.load("./model_sub/head.pt", map_location="cpu"))
net = net.to(DEV).eval()

preds = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt").to(DEV)
        m = torch.tensor(Mt[i:i + 64]).to(DEV)
        cls = net.backbone(**b).last_hidden_state[:, 0]
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

zp = "submit_hybrid2.zip"
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
print("1GB 초과 시 MODEL_NAME 축소 or fp16 저장 필요.")
