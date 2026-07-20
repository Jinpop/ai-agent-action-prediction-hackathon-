"""Colab GPU 용 다국어 트랜스포머 파인튜닝 (Dacon 236694).

[ 실행법 (Google Colab, 런타임 > GPU(T4) 선택) ]
  1) Dacon에서 받은 open.zip(안에 data/ 폴더) 를 Colab 세션에 업로드
     또는 data/ 의 train.jsonl, train_labels.csv, test.jsonl, sample_submission.csv 업로드
  2) 이 파일 업로드 후, 셀에서:
        !pip -q install "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece scikit-learn
        !python colab_train_transformer.py
  3) 끝나면 submit_transformer.zip 이 생성됨 -> 다운로드해서 Dacon 제출

산출물:
  - ./model/  (파인튜닝된 모델+토크나이저)
  - ./submit_transformer.zip  (script.py + requirements.txt + model/)  <- 제출
홀드아웃 Macro-F1(세션 단위)과 클래스별 F1을 출력한다.
"""
import json
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# ================== 설정 ==================
MODEL_NAME = os.environ.get("MODEL_NAME", "klue/roberta-base")
# 한/영 혼용이 심하면 "xlm-roberta-base" 로 바꿔 실험(단 저장크기 커서 fp16 저장 권장).
DATA_DIR = "./data"
OUT_DIR = "./model"
MAX_LEN = int(os.environ.get("MAX_LEN", 160))
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH = int(os.environ.get("BATCH", 32))
LR = 2e-5
SEED = 42
N_LIMIT = int(os.environ.get("N_LIMIT", 0))      # >0 이면 학습 데이터 subset (smoke test)
SKIP_REFIT = os.environ.get("SKIP_REFIT", "0") == "1"

ACTIONS = [
    "edit_file", "grep_search", "read_file", "glob_pattern", "respond_only",
    "run_bash", "apply_patch", "run_tests", "list_directory", "ask_user",
    "plan_task", "lint_or_typecheck", "write_file", "web_search",
]


# ---------- 입력 텍스트 (상태 토큰 포함) ----------
def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def build_text(sample):
    parts = []
    hist = sample.get("history") or []
    for h in hist:
        if isinstance(h, dict) and h.get("role") == "user":
            parts.append(_s(h.get("content", "")))
    parts = parts[-3:]
    acts = [h.get("name") for h in hist
            if isinstance(h, dict) and h.get("role") == "assistant_action"]
    for a in acts[-8:]:
        parts.append("seq_" + _s(a))
    ws = (sample.get("session_meta") or {}).get("workspace") or {}
    tier = _s((sample.get("session_meta") or {}).get("user_tier"))
    ci = _s(ws.get("last_ci_status"))
    parts.append(f"tier_{tier} ci_{ci} turn_{(sample.get('session_meta') or {}).get('turn_index', 0)}")
    for p in (ws.get("open_files") or [])[:6]:
        parts.append("openf_" + _s(p).replace("/", "_").replace(".", "_"))
    parts.append(_s(sample.get("current_prompt", "")))
    return " ".join(parts).strip()


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def session_of(_id):
    return _id.rsplit("-step", 1)[0]


# ---------- 데이터 로드 ----------
print(f"Model: {MODEL_NAME}")
train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
labels_df = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
lab = dict(zip(labels_df["id"], labels_df["action"]))

lab2id = {a: i for i, a in enumerate(ACTIONS)}
id2lab = {i: a for a, i in lab2id.items()}

if N_LIMIT:
    train = train[:N_LIMIT]
    print(f"[smoke] subset {N_LIMIT} samples")
ids = [s["id"] for s in train]
texts = [build_text(s) for s in train]
y = np.array([lab2id[lab[i]] for i in ids])
groups = np.array([session_of(i) for i in ids])

tr, va = next(GroupKFold(n_splits=5).split(texts, y, groups))
print(f"train={len(tr)}  holdout={len(va)}")

# 클래스 가중치(불균형 -> 가중 손실)
counts = np.bincount(y[tr], minlength=len(ACTIONS)).astype(np.float64)
cls_w = (counts.sum() / (len(ACTIONS) * np.maximum(counts, 1))).astype(np.float32)
class_weights = torch.tensor(cls_w)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def make_ds(idx):
    d = Dataset.from_dict({"text": [texts[i] for i in idx],
                           "label": [int(y[i]) for i in idx]})
    return d.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                 batched=True, remove_columns=["text"])


ds_tr, ds_va = make_ds(tr), make_ds(va)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(ACTIONS), id2label=id2lab, label2id=lab2id)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            out.logits, labels, weight=class_weights.to(out.logits.device))
        return (loss, out) if return_outputs else loss


def metrics(p):
    pred = p.predictions.argmax(-1)
    return {"macro_f1": f1_score(p.label_ids, pred, average="macro")}


args = TrainingArguments(
    output_dir="./_ckpt", num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
    learning_rate=LR, warmup_ratio=0.1, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="no", logging_steps=100,
    fp16=torch.cuda.is_available(), seed=SEED, report_to="none",
)
trainer = WeightedTrainer(model=model, args=args, train_dataset=ds_tr,
                          eval_dataset=ds_va, compute_metrics=metrics,
                          processing_class=tok)
trainer.train()

# ---------- 홀드아웃 평가 ----------
pred = trainer.predict(ds_va).predictions.argmax(-1)
f1 = f1_score(y[va], pred, average="macro")
print(f"\n==> 홀드아웃 Macro-F1 = {f1:.4f}")
print(classification_report(y[va], pred, target_names=ACTIONS, digits=3, zero_division=0))

# ---------- 전체 재학습(제출용) ----------
if SKIP_REFIT:
    print("[smoke] SKIP_REFIT=1 -> 홀드아웃 모델을 그대로 저장(재학습 생략)")
    model_full = model
else:
    print("\nRefit on FULL data...")
    ds_all = make_ds(np.arange(len(texts)))
    model_full = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(ACTIONS), id2label=id2lab, label2id=lab2id)
    counts_all = np.bincount(y, minlength=len(ACTIONS)).astype(np.float64)
    class_weights = torch.tensor(
        (counts_all.sum() / (len(ACTIONS) * np.maximum(counts_all, 1))).astype(np.float32))
    args.eval_strategy = "no"   # 전체 재학습엔 검증셋 없음
    trainer_full = WeightedTrainer(model=model_full, args=args, train_dataset=ds_all,
                                   processing_class=tok)
    trainer_full.train()

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
model_full.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"saved model -> {OUT_DIR}")

# ---------- 제출용 script.py / requirements.txt 작성 ----------
SUBMIT_SCRIPT = r'''"""추론(제출용) — 파인튜닝 트랜스포머로 test.jsonl 예측."""
import json, os, csv
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ACTIONS = ["edit_file","grep_search","read_file","glob_pattern","respond_only",
           "run_bash","apply_patch","run_tests","list_directory","ask_user",
           "plan_task","lint_or_typecheck","write_file","web_search"]
def _s(x): return x if isinstance(x,str) else ("" if x is None else str(x))
def build_text(sample):
    parts=[]; hist=sample.get("history") or []
    for h in hist:
        if isinstance(h,dict) and h.get("role")=="user": parts.append(_s(h.get("content","")))
    parts=parts[-3:]
    acts=[h.get("name") for h in hist if isinstance(h,dict) and h.get("role")=="assistant_action"]
    for a in acts[-8:]: parts.append("seq_"+_s(a))
    ws=(sample.get("session_meta") or {}).get("workspace") or {}
    tier=_s((sample.get("session_meta") or {}).get("user_tier")); ci=_s(ws.get("last_ci_status"))
    parts.append(f"tier_{tier} ci_{ci} turn_{(sample.get('session_meta') or {}).get('turn_index',0)}")
    for p in (ws.get("open_files") or [])[:6]: parts.append("openf_"+_s(p).replace("/","_").replace(".","_"))
    parts.append(_s(sample.get("current_prompt",""))); return " ".join(parts).strip()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model").to(DEV).eval()
samples=[json.loads(l) for l in open("./data/test.jsonl",encoding="utf-8") if l.strip()]
ids=[s.get("id","") for s in samples]; texts=[build_text(s) for s in samples]
preds=[]
with torch.no_grad():
    for i in range(0,len(texts),64):
        b=tok(texts[i:i+64],truncation=True,max_length=160,padding=True,return_tensors="pt").to(DEV)
        preds.extend(model(**b).logits.argmax(-1).cpu().numpy().tolist())
pm={i:ACTIONS[p] for i,p in zip(ids,preds)}
os.makedirs("./output",exist_ok=True)
with open("./data/sample_submission.csv",newline="",encoding="utf-8") as f:
    r=csv.DictReader(f); fields=r.fieldnames; rows=list(r)
for row in rows:
    if row["id"] in pm: row["action"]=pm[row["id"]]
with open("./output/submission.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print("saved ./output/submission.csv", len(rows))
'''
open("script.py", "w", encoding="utf-8").write(SUBMIT_SCRIPT)
open("requirements.txt", "w", encoding="utf-8").write(
    "transformers==4.44.2\ntorch\nsentencepiece\n")

# ---------- 제출 zip ----------
zp = "submit_transformer.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("script.py")
    z.write("requirements.txt")
    for root, _, files in os.walk(OUT_DIR):
        for fn in files:
            fp = os.path.join(root, fn)
            z.write(fp, os.path.relpath(fp, "."))
size = os.path.getsize(zp) / 1e6
print(f"\n[완료] {zp}  ({size:.0f} MB)  홀드아웃 Macro-F1={f1:.4f}")
print("1GB 초과면 MODEL_NAME 을 더 작은 모델로 바꾸거나 fp16 저장 필요.")
