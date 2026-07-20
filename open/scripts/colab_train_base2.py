"""Kaggle/Colab GPU 용 v17 멤버 — base 하이브리드 개선판 (au 인지 + MAX_LEN 512 + seed 43).

v12(base, ~0.70) 대비 변경 3가지:
  * [SRC] au|sim 프리픽스 — sess_au_* 세션(7.2%)은 라벨 분포가 다른 별개 집단인데
    기존 모델은 집단을 인지 못 함(v14가 au에서 F1 0.51). 텍스트로 명시.
  * MAX_LEN 384 -> 512 — 트랜스크립트 p90=553토큰, 384에선 44%가 잘림.
  * SEED 43 — v12(seed 42)와 앙상블 다양성 확보.

large 교훈: 서버 T4 추론 600초 안에 large 완주 불가(LB 0.67로 희석) → base 2개
앙상블이 배포 가능한 최적 전략. base fp16+정렬 추론 ~170초(512는 ~230초).

[ Kaggle 실행법 (GPU, 단일 GPU 강제) ]
  BASE = "/kaggle/input/datasets/jinpop/dacon236694"
  !cp {BASE}/feat.py {BASE}/colab_train_base2.py .
  !mkdir -p data && cp {BASE}/colab_data/* data/
  !pip -q install "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece scikit-learn joblib
  !CUDA_VISIBLE_DEVICES=0 BATCH=48 python colab_train_base2.py     # 512라 배치 48
  (T4 단일 총 ~3.5시간. 끝나면 submit_base2.zip / holdout_probs3.npy / holdout_idx3.npy 다운로드)

주의: 로컬에서 import 금지(즉시 학습 시작됨).
"""
import json
import os
import shutil
import sys
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
from transformers import (AutoModel, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)
from transformers.modeling_outputs import SequenceClassifierOutput

import feat  # 같이 업로드 필수

# ================== 설정 ==================
MODEL_NAME = os.environ.get("MODEL_NAME", "klue/roberta-base")
DATA_DIR = "./data"
SUB_DIR = "./model_sub"
MAX_LEN = int(os.environ.get("MAX_LEN", 512))
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH = int(os.environ.get("BATCH", 48))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 1))
LR = float(os.environ.get("LR", 2e-5))
SEED = int(os.environ.get("SEED", 43))
N_LIMIT = int(os.environ.get("N_LIMIT", 0))
SKIP_REFIT = os.environ.get("SKIP_REFIT", "0") == "1"
REFIT_ONLY = os.environ.get("REFIT_ONLY", "0") == "1"   # 홀드아웃 생략, 전체 재학습만(복구용)
HOLDOUT_ONLY = os.environ.get("HOLDOUT_ONLY", "0") == "1"  # 홀드아웃 학습·평가·밴드입력(npy) 저장 후 GATE_DONE 기록하고 즉시 종료. full-refit/model_sub/script.py/requirements/zip 전부 생략(밴드 자격만 계산용).
COARSE_AUX = float(os.environ.get("COARSE_AUX", "0"))   # >0 = 4행동군 보조 CE 가중(λ). 배포 head 무변경.
PRETEXT = os.environ.get("PRETEXT", "0") == "1"         # 1단계: EXTRA(mint)만 text-only(메타 0화) 학습 → backbone 저장 후 종료
PRETEXT_META = os.environ.get("PRETEXT_META", "blank")  # blank=session_meta 제거(빈 META 상수) / omit=[META]라인 생략(A/B 대조군) / keep=데이터 그대로(A/B 실험군: exact 필드)
INIT_BACKBONE = os.environ.get("INIT_BACKBONE", "")     # 2단계: 사전학습 backbone 경로에서 초기화(real-only 권장)
MASK_MINT_META = os.environ.get("MASK_MINT_META", "0") == "1"
# ★META wave (campaign official-meta-wave1): META-N=숫자메타 +6키(feat.py가 사용) / META-T=transcript 전체 공식메타.
META_NUM_EXT = os.environ.get("META_NUM_EXT", "0") == "1"      # feat.py 가 실제 사용(여기선 effective_config 기록용)
META_TRANS_EXT = os.environ.get("META_TRANS_EXT", "0") == "1"  # build_transcript [META]에 전체 공식 메타 추가
# ★train/serve skew 가드(campaign official-meta-wave1 #2): feat 은 import 시점에 env 를 읽으므로
# colab 파싱값과 반드시 동치여야 함. import feat(상단) 이후 env 를 바꾸면 여기서 조용한 119d 오작동을 잡는다.
assert META_NUM_EXT == feat.META_NUM_EXT, \
    f"META_NUM_EXT 불일치(colab={META_NUM_EXT} feat={feat.META_NUM_EXT}) — feat import 전에 env 설정 필요"
assert PRETEXT_META in ("blank", "omit", "keep")
assert PRETEXT or PRETEXT_META == "blank", "PRETEXT_META는 PRETEXT=1 전용"
assert not (PRETEXT and MASK_MINT_META), \
    "PRETEXT와 MASK_MINT_META 동시 지정 금지 — keep 모드의 exact dirty/open이 조용히 삭제됨(07-11 검수 지적)"
assert not (SKIP_REFIT and REFIT_ONLY), "SKIP_REFIT과 REFIT_ONLY 동시 지정 불가"
assert not (HOLDOUT_ONLY and REFIT_ONLY), "HOLDOUT_ONLY와 REFIT_ONLY 동시 지정 불가 — HOLDOUT_ONLY는 홀드아웃 branch(REFIT_ONLY의 else)가 필요"
assert not (PRETEXT and INIT_BACKBONE), "PRETEXT와 INIT_BACKBONE 동시 지정 불가"
assert not PRETEXT or os.environ.get("EXTRA_DATA", ""), "PRETEXT=1은 EXTRA_DATA(mint) 필수"
GRAD_CKPT = os.environ.get("GRAD_CKPT", "0") == "1"
SIBLING_ONLY = os.environ.get("SIBLING_ONLY", "0") == "1"  # 4형제 전용 분류기 모드
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", 0.0))  # soft타깃 저비용 근사
_SOFT_PATH = os.environ.get("SOFT_TARGET", "")             # 자기증류 2단계 soft label(N,14 확률)
DISTILL_T = float(os.environ.get("DISTILL_T", 2.0))         # 증류 온도
DISTILL_ALPHA = float(os.environ.get("DISTILL_ALPHA", 0.5)) # 하드CE 비중(1-alpha가 KD)
EXTRA_DATA = os.environ.get("EXTRA_DATA", "")               # 민트 증강 jsonl (라벨은 _labels.csv)
SESSION_EQUAL = os.environ.get("SESSION_EQUAL", "0") == "1" # 세션 균등가중(서버=세션당1스텝 정렬)
AU_BOOST = float(os.environ.get("AU_BOOST", 1.0))           # au 세션 가중 배수
TARGET_BALANCE = os.environ.get("TARGET_BALANCE", "0") == "1"  # mint2 balanced: target_key별 총가중=1 (window 과표집 제거)
TB_V2 = os.environ.get("TB_V2", "0") == "1"  # 07-14 외부감사 수정: ①배치독립 가중손실((per*wt).mean, WT 전역 mean=1 전제 — 기존 배치별 wt.sum() 재정규화는 target 기여가 배치구성 의존, 재현 max/min≈1.9배) ②클래스가중을 target가중 반영으로(bincount weights=WT — 기존 raw 행수는 rolling/exact 간 multiplier 0.70~0.98배 요동)
HOLDOUT_IDX = os.environ.get("HOLDOUT_IDX", "")            # 고정 홀드아웃 idx npy(hidx). 미지정시 GroupKFold-fold0

if SIBLING_ONLY:
    # 탐색 4형제만 4-way 분류 (홀드아웃 40% 광맥에 용량 집중)
    ACTIONS = ["read_file", "grep_search", "glob_pattern", "list_directory"]
    print("[SIBLING] 4형제 전용 모드 — 4-way 분류")
else:
    ACTIONS = feat.ACTIONS
lab2id = {a: i for i, a in enumerate(ACTIONS)}
# 사전고정 4 행동군(coarse family): 탐색0 / 추론실행1 / 편집2 / 응답3 — COARSE_AUX용
COARSE_FAMILY = {
    "read_file": 0, "grep_search": 0, "glob_pattern": 0, "list_directory": 0,
    "run_bash": 1, "run_tests": 1, "lint_or_typecheck": 1, "web_search": 1,
    "plan_task": 1, "ask_user": 1,
    "edit_file": 2, "apply_patch": 2, "write_file": 2,
    "respond_only": 3,
}


def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def build_transcript(sample):
    """v12 직렬화 + [SRC] 집단 프리픽스 (추론 script.py 와 반드시 동일)."""
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
    if PRETEXT_META != "omit":  # omit = [META] 라인 자체 생략 (PRETEXT 전용 — 배포 경로에선 항상 포함)
        meta = sample.get("session_meta") or {}
        ws = meta.get("workspace") or {}
        if META_TRANS_EXT:  # META-T: 전체 공식 메타를 [META] transcript 에 추가(숫자메타는 불변)
            lm = ws.get("language_mix") or {}
            lm_str = ",".join(f"{k}:{lm[k]}" for k in sorted(lm))
            parts.append(
                f"[META] tier={_s(meta.get('user_tier'))} langpref={_s(meta.get('language_pref'))} "
                f"ci={_s(ws.get('last_ci_status'))} dirty={int(bool(ws.get('git_dirty')))} "
                f"turn={meta.get('turn_index', 0)} budget={meta.get('budget_tokens_remaining', 0)} "
                f"elapsed={meta.get('elapsed_session_sec', 0)} loc={ws.get('loc', 0)} "
                f"langmix={lm_str} "
                f"open={','.join(_s(p) for p in (ws.get('open_files') or [])[:6])}")
        else:
            parts.append(
                f"[META] tier={_s(meta.get('user_tier'))} ci={_s(ws.get('last_ci_status'))} "
                f"dirty={int(bool(ws.get('git_dirty')))} turn={meta.get('turn_index', 0)} "
                f"open={','.join(_s(p) for p in (ws.get('open_files') or [])[:6])}")
    parts.append("[P] " + _s(sample.get("current_prompt", "")))
    return "\n".join(parts)


# ---------- effective config 기록 (07-12 교정10: printenv 금지 — 파싱된 값만, 기본값 포함) ----------
def _sha256_of(path):
    try:
        import hashlib
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def _dump_effective_config():
    import transformers as _tf
    import sklearn as _sk
    cfg = {
        "MODEL_NAME": MODEL_NAME, "MAX_LEN": MAX_LEN, "EPOCHS": EPOCHS, "BATCH": BATCH,
        "GRAD_ACCUM": GRAD_ACCUM, "LR": LR, "SEED": SEED, "GRAD_CKPT": GRAD_CKPT,
        "PRETEXT": PRETEXT, "PRETEXT_META": PRETEXT_META, "MASK_MINT_META": MASK_MINT_META,
        "META_NUM_EXT": META_NUM_EXT, "META_TRANS_EXT": META_TRANS_EXT,
        "EXTRA_DATA": EXTRA_DATA, "INIT_BACKBONE": INIT_BACKBONE, "COARSE_AUX": COARSE_AUX,
        "LABEL_SMOOTH": LABEL_SMOOTH, "SESSION_EQUAL": SESSION_EQUAL, "AU_BOOST": AU_BOOST,
        "TARGET_BALANCE": TARGET_BALANCE, "TB_V2": TB_V2, "HOLDOUT_IDX": HOLDOUT_IDX,
        "SOFT_TARGET": _SOFT_PATH, "SKIP_REFIT": SKIP_REFIT, "REFIT_ONLY": REFIT_ONLY,
        "HOLDOUT_ONLY": HOLDOUT_ONLY,
        "SIBLING_ONLY": SIBLING_ONLY, "N_LIMIT": N_LIMIT,
        "DISTILL_T": DISTILL_T, "DISTILL_ALPHA": DISTILL_ALPHA,
        "sha256": {
            "colab_train_base2.py": _sha256_of(os.path.abspath(__file__)),
            "feat.py": _sha256_of(feat.__file__),
            "train.jsonl": _sha256_of(os.path.join(DATA_DIR, "train.jsonl")),
            "EXTRA_DATA": _sha256_of(EXTRA_DATA) if EXTRA_DATA else None,
            "EXTRA_labels": _sha256_of(EXTRA_DATA.replace(".jsonl", "_labels.csv")) if EXTRA_DATA else None,
            "INIT_BACKBONE_weights": _sha256_of(os.path.join(INIT_BACKBONE, "model.safetensors")) if INIT_BACKBONE else None,
            "HOLDOUT_IDX": _sha256_of(HOLDOUT_IDX) if HOLDOUT_IDX else None,
        },
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "cuda": torch.version.cuda, "transformers": _tf.__version__,
                     "sklearn": _sk.__version__, "numpy": np.__version__},
    }
    with open("effective_config.json", "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    print("[CFG] effective_config.json 저장 (파싱값·기본값·SHA·버전)")


_dump_effective_config()

# ---------- 데이터 로드 ----------
print(f"Model: {MODEL_NAME}  (v17 base2: [SRC]+512, seed={SEED})")
train = feat.load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
labels_df = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
lab = dict(zip(labels_df["id"], labels_df["action"]))
if SIBLING_ONLY:
    _sib = set(ACTIONS)
    train = [s for s in train if lab.get(s["id"]) in _sib]
    print(f"[SIBLING] 4형제 샘플만: {len(train)}행")
if N_LIMIT:
    train = train[:N_LIMIT]
    print(f"[smoke] subset {N_LIMIT}")

N_REAL = len(train)   # 분할은 실제 행에서만; EXTRA(민트)는 학습쪽에만 편입
if EXTRA_DATA:
    extra = feat.load_jsonl(EXTRA_DATA)
    ex_lab = pd.read_csv(EXTRA_DATA.replace(".jsonl", "_labels.csv"))
    lab.update(dict(zip(ex_lab["id"], ex_lab["action"])))
    if MASK_MINT_META:
        # 시점 복원 불가능한 메타를 mint 행에서 제거(오염신호 학습 차단). 텍스트 [META]와
        # 119메타 벡터 양쪽에 일관 반영되도록 소스 dict에서 키 자체를 삭제.
        # ci는 EXTRA_DATA로 복원본(train_mint_ci)을 쓰면 유지됨. turn_index는 민팅시 조정돼 유지.
        for _s_ in extra:
            _m = _s_.get("session_meta") or {}
            _w = _m.get("workspace") or {}
            _w.pop("open_files", None); _w.pop("git_dirty", None)
            _m.pop("budget_tokens_remaining", None); _m.pop("elapsed_session_sec", None)
        print(f"[MASK] mint 메타 마스킹: open_files/git_dirty/budget/elapsed 제거 ({len(extra)}행)")
    if PRETEXT:  # 1단계: mint만으로 text-only 사전학습 (real train 미사용)
        # ★text-only 완전화(07-11 교정9): 119벡터 0화만으론 부족 — transcript [META] 문자열로
        # 오염 메타가 새므로 blank 모드는 session_meta 자체를 제거(전행 동일 빈 META = 무신호).
        # omit 모드는 build_transcript에서 [META] 라인 생략, keep 모드는 데이터의 exact 필드 유지(A/B 실험군).
        if PRETEXT_META == "blank":
            for _s_ in extra:
                _s_.pop("session_meta", None)
        train = extra
        N_REAL = len(train)
        print(f"[PRETEXT] mint-only 모드: {len(train)}행 (real 제외, META처리={PRETEXT_META}, 119벡터 0화)")
    else:
        train = train + extra
        print(f"[EXTRA] 민트 {len(extra)}행 추가 (분할은 실제 {N_REAL}행 기준)")

ids = [s["id"] for s in train]
texts = [build_transcript(s) for s in train]
y = np.array([lab2id[lab[i]] for i in ids])
groups = np.array([feat.session_of(i) for i in ids])
target_keys = [s.get("target_key") for s in train]   # mint2 balanced 전용(그 외 None)

# 학습 샘플 가중치 (세션균등 / au부스트) — 손실에 곱해짐
WT = np.ones(len(ids), dtype=np.float32)
if SESSION_EQUAL:
    from collections import Counter as _C
    _cnt = _C(groups)
    WT *= np.array([1.0 / _cnt[g] for g in groups], dtype=np.float32)
if AU_BOOST != 1.0:
    _au = np.array([i.startswith("sess_au") for i in ids])
    WT[_au] *= AU_BOOST
if TARGET_BALANCE:
    # mint2 balanced: canonical target별 총가중=1 (window N개면 각 1/N). target_key는 빌더가 emit.
    from collections import Counter as _TC
    assert all(t is not None for t in target_keys), \
        "TARGET_BALANCE=1은 EXTRA_DATA 전행에 target_key 필드 필수 (balanced 빌더 산출만 허용)"
    _tcnt = _TC(target_keys)
    WT *= np.array([1.0 / _tcnt[t] for t in target_keys], dtype=np.float32)
    print(f"[WT] target_balance: {len(_tcnt)} target, window {len(ids)}행 → 총가중 target당 1")
WT /= WT.mean()
if SESSION_EQUAL or AU_BOOST != 1.0 or TARGET_BALANCE:
    print(f"[WT] 세션균등={SESSION_EQUAL} au×{AU_BOOST} tb={TARGET_BALANCE} (w범위 {WT.min():.3f}~{WT.max():.3f})")

# 자기증류 soft target 로드 (SOFT_TARGET=경로). oof_ids로 현재 train 순서에 정렬.
SOFT = None
if _SOFT_PATH:
    _sd = np.load(_SOFT_PATH)  # dict-like npz: 'probs'(N,14) + 'ids'(N) + 'temperature'
    if "temperature" in getattr(_sd, "files", []):   # ★07-13 v26 버그방지: teacher/student KD 온도 일치 강제
        _sT = float(_sd["temperature"])
        assert abs(_sT - DISTILL_T) < 1e-6, \
            f"soft npz temperature {_sT} != DISTILL_T {DISTILL_T} — teacher/student KD 온도 불일치(v26 버그). make_soft_target --T {DISTILL_T}로 재생성"
        print(f"[DISTILL] temperature matched: teacher T={_sT} == DISTILL_T")
    _probs, _sids = _sd["probs"], _sd["ids"]  # npz lazy 접근은 1회만 (반복 접근 금물)
    _sid2row = {i: r for r, i in enumerate(_sids)}
    SOFT = _probs[[_sid2row[i] for i in ids]].astype(np.float32)
    print(f"[DISTILL] soft target 로드: {SOFT.shape}, T={DISTILL_T} alpha={DISTILL_ALPHA}")

print("Build meta features...")
meta_df = feat.build_meta_frame(train)
META_COLS = list(meta_df.columns)
M = meta_df.values.astype(np.float32)
if PRETEXT:
    M[:] = 0.0   # text-only: 메타 신호 미전달 (스케일러는 0분산→scale 1, 0 유지)
    print("  [PRETEXT] meta 전체 0화")
print(f"  meta dims = {M.shape[1]}")

if HOLDOUT_IDX:
    # 고정 홀드아웃(hidx): 기존 멤버 홀드아웃 로짓이 전부 이 split이라, 밴드체크 정렬·누수0을 위해 pin.
    va = np.load(HOLDOUT_IDX).astype(int).ravel()
    assert va.ndim == 1 and 0 <= va.min() and va.max() < N_REAL, "HOLDOUT_IDX가 real[0:N_REAL] 범위를 벗어남"
    tr = np.setdiff1d(np.arange(N_REAL), va)
    print(f"[HOLDOUT] 고정 홀드아웃 {HOLDOUT_IDX}: va={len(va)} tr={len(tr)} (GroupKFold-fold0 대체)")
else:
    tr, va = next(GroupKFold(n_splits=5).split(texts[:N_REAL], y[:N_REAL], groups[:N_REAL]))
if len(texts) > N_REAL:  # 민트는 홀드아웃 세션 제외하고 학습쪽에만
    _va_sess = set(groups[va])
    _ok = np.array([i for i in range(N_REAL, len(texts)) if groups[i] not in _va_sess])
    tr = np.concatenate([tr, _ok.astype(tr.dtype)])
    print(f"[EXTRA] 학습 편입 {len(_ok)} / 홀드아웃세션이라 제외 {len(texts)-N_REAL-len(_ok)}")
print(f"train={len(tr)}  holdout={len(va)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.truncation_side = "left"   # [META]/[P] 보존, 오래된 history 부터 자름
# 주의: [SRC] 프리픽스는 맨 앞이라 512 초과 초장문에선 잘릴 수 있으나(1% 미만),
# au 신호는 메타러너의 au 플래그로도 이중 공급되므로 허용.


def make_ds(idx, scaler):
    Ms = scaler.transform(M[idx]).astype(np.float32)
    cols = {
        "text": [texts[i] for i in idx],
        "label": [int(y[i]) for i in idx],
        "meta": [row.tolist() for row in Ms],
    }
    if SOFT is not None:
        cols["soft"] = [SOFT[i].tolist() for i in idx]
    if SESSION_EQUAL or AU_BOOST != 1.0 or TARGET_BALANCE:
        cols["wt"] = [float(WT[i]) for i in idx]
    d = Dataset.from_dict(cols)
    return d.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                 batched=True, remove_columns=["text"])


def sqrt_weights(yy, wt=None):
    # TB_V2(07-14): wt 주어지면 클래스 카운트를 가중 행수(= target 등가 수)로 — raw window 행수는
    # rolling/exact 간 클래스 multiplier를 0.70~0.98배 요동시킴(외부감사 지적).
    if wt is not None:
        counts = np.bincount(yy, weights=np.asarray(wt, dtype=np.float64),
                             minlength=len(ACTIONS)).astype(np.float64)
    else:
        counts = np.bincount(yy, minlength=len(ACTIONS)).astype(np.float64)
    w = np.sqrt(counts.sum() / (len(ACTIONS) * np.maximum(counts, 1)))
    return torch.tensor((w / w.mean()).astype(np.float32))


def extend_positions(backbone, new_max):
    """roberta 위치임베딩 확장 (512 하드제한 해제). 기존 512위치를 선형보간으로
    new_max-2 위치로 늘림 — 파인튜닝으로 적응. pos 0,1(pad슬롯)은 보존."""
    emb = getattr(backbone.embeddings, "position_embeddings", None)
    if emb is None:  # 상대위치 모델(DeBERTa 계열)은 절대위치 테이블이 없음 — 확장 불필요
        return
    old_n, dim = emb.weight.shape
    if new_max <= old_n:
        return
    w = emb.weight.data
    core = w[2:].t().unsqueeze(0)                      # (1, dim, old_n-2)
    new_core = F.interpolate(core, size=new_max - 2, mode="linear",
                             align_corners=True).squeeze(0).t()
    new_emb = nn.Embedding(new_max, dim, padding_idx=emb.padding_idx)
    new_emb.weight.data = torch.cat([w[:2], new_core], 0)
    backbone.embeddings.position_embeddings = new_emb
    dev = w.device
    if hasattr(backbone.embeddings, "position_ids"):
        backbone.embeddings.register_buffer(
            "position_ids", torch.arange(new_max, device=dev).expand(1, -1), persistent=False)
    if hasattr(backbone.embeddings, "token_type_ids"):
        backbone.embeddings.register_buffer(
            "token_type_ids", torch.zeros((1, new_max), dtype=torch.long, device=dev),
            persistent=False)
    backbone.config.max_position_embeddings = new_max
    print(f"[POS] 위치임베딩 {old_n} -> {new_max} 확장 (MAX_LEN {new_max-2})")


class HybridNet(nn.Module):
    def __init__(self, name, n_meta, n_labels, class_weights=None):
        super().__init__()
        _bb_src = INIT_BACKBONE or name   # 2단계: 사전학습 backbone에서 초기화
        if INIT_BACKBONE:
            print(f"[INIT] backbone <- {INIT_BACKBONE}")
        self.backbone = AutoModel.from_pretrained(_bb_src)
        if MAX_LEN + 2 > self.backbone.config.max_position_embeddings:
            extend_positions(self.backbone, MAX_LEN + 2)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h + n_meta, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_labels))
        self.register_buffer("cw", class_weights if class_weights is not None
                             else torch.ones(n_labels))
        if COARSE_AUX > 0:  # 보조 head는 head.pt(=self.head)에 저장 안 됨 → 배포 계약 무변경
            self.aux_head = nn.Linear(256, 4)
            self.register_buffer("fam_map", torch.tensor(
                [COARSE_FAMILY[a] for a in ACTIONS], dtype=torch.long))

    def forward(self, input_ids=None, attention_mask=None, meta=None,
                labels=None, soft=None, wt=None, **kwargs):
        cls = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state[:, 0]
        feats = torch.cat([cls, meta.float()], dim=-1)
        if COARSE_AUX > 0:  # 256 병목 공유(동일 dropout 마스크), main/aux 두 head
            z = self.head[2](self.head[1](self.head[0](feats)))
            logits = self.head[3](z)
        else:
            logits = self.head(feats)
        loss = None
        if labels is not None:
            if wt is not None:                         # 샘플 가중 CE (세션균등/au부스트)
                per = F.cross_entropy(logits, labels, weight=self.cw,
                                      reduction="none", label_smoothing=LABEL_SMOOTH)
                if TB_V2:
                    # 배치독립 가중손실(07-14): WT 전역 mean=1(:234) 전제, 행 기여 = wt_i/B로
                    # 배치 구성과 무관하게 고정. 기존 sum/wt.sum()은 배치별 재정규화로
                    # target 기여가 함께 들어간 배치에 의존(재현 max/min≈1.9배).
                    loss = (per * wt.float()).mean()
                else:
                    loss = (per * wt.float()).sum() / wt.float().sum().clamp(min=1e-8)
            else:
                loss = F.cross_entropy(logits, labels, weight=self.cw,
                                       label_smoothing=LABEL_SMOOTH)
            if soft is not None:                       # 자기증류: 하드CE + KD 혼합
                T = DISTILL_T
                kd = F.kl_div(F.log_softmax(logits / T, dim=-1),
                              soft.float(), reduction="batchmean") * (T * T)
                loss = DISTILL_ALPHA * loss + (1 - DISTILL_ALPHA) * kd
            if COARSE_AUX > 0:                         # 4행동군 보조 CE (배포 시 head 폐기)
                loss = loss + COARSE_AUX * F.cross_entropy(
                    self.aux_head(z), self.fam_map[labels])
        return SequenceClassifierOutput(loss=loss, logits=logits)


def metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}


def train_model(idx_tr, idx_va=None):
    scaler = StandardScaler().fit(M[idx_tr])
    ds_tr = make_ds(idx_tr, scaler)
    ds_va = make_ds(idx_va, scaler) if idx_va is not None else None
    set_seed(SEED)   # 재현성 교정(07-11 §9): 모델 생성 전 시드 고정 — head 초기화까지 seed 제어
    model = HybridNet(MODEL_NAME, M.shape[1], len(ACTIONS),
                      sqrt_weights(y[idx_tr], WT[idx_tr] if TB_V2 else None))
    if GRAD_CKPT:
        model.backbone.gradient_checkpointing_enable()
    args = TrainingArguments(
        output_dir="./_ckpt", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=32,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch" if ds_va is not None else "no",
        save_strategy="no", logging_steps=100,
        fp16=torch.cuda.is_available(), seed=SEED, report_to="none",
        remove_unused_columns=True,  # soft/wt는 forward 시그니처에 있어 보존됨
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr,
                      eval_dataset=ds_va, compute_metrics=metrics if ds_va is not None else None,
                      processing_class=tok)
    trainer.train()
    return trainer, model, scaler


# ---------- PRETEXT 모드 (2단계 학습의 1단계: mint text-only 사전학습) ----------
if PRETEXT:
    print(f"[PRETEXT] {len(texts)}행 text-only 학습 ({EPOCHS}ep) -> backbone만 저장")
    _, m_pre, _ = train_model(np.arange(len(texts)))
    PRE_DIR = "./pretext_backbone"
    if os.path.exists(PRE_DIR):
        shutil.rmtree(PRE_DIR)
    m_pre.backbone.float().save_pretrained(PRE_DIR)   # fp32 저장(2단계 학습 초기화용)
    print(f"[PRETEXT] saved -> {PRE_DIR}")
    sys.exit(0)


# ---------- OOF 모드 (자기증류 1단계: soft label 생성) ----------
# OOF=1: 5-fold GroupKFold로 각 fold를 held로 학습·예측 → train 전체 out-of-fold logit.
# 여러 SEED로 반복해 평균하면 앙상블 soft target. refit/zip 없이 종료.
if os.environ.get("OOF", "0") == "1":
    import gc as _gc
    from sklearn.model_selection import GroupKFold as _GKF
    oof = np.full((len(texts), len(ACTIONS)), np.nan, np.float32)
    for k, (tr_k, va_k) in enumerate(_GKF(5).split(texts, y, groups)):
        print(f"[OOF] fold {k+1}/5  train={len(tr_k)} held={len(va_k)}", flush=True)
        tr_k_obj, m_k, sc_k = train_model(tr_k, None)  # eval 생략(속도)
        oof[va_k] = tr_k_obj.predict(make_ds(va_k, sc_k)).predictions
        del tr_k_obj, m_k
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert not np.isnan(oof).any(), "OOF 커버리지 불완전"
    np.save(f"oof_logits_seed{SEED}.npy", oof)
    with open(f"oof_ids_seed{SEED}.json", "w") as f:
        json.dump(ids, f)
    print(f"[OOF] 저장: oof_logits_seed{SEED}.npy {oof.shape}", flush=True)
    sys.exit(0)


# ---------- 홀드아웃 학습/평가 ----------
if REFIT_ONLY:
    print("[REFIT_ONLY] 홀드아웃 단계 생략 -> 전체 재학습만 (복구/재생성용)")
else:
    trainer, model, scaler = train_model(tr, va)
    out = trainer.predict(make_ds(va, scaler))
    pred = out.predictions.argmax(-1)
    f1 = f1_score(y[va], pred, average="macro")
    print(f"\n==> 홀드아웃 Macro-F1 = {f1:.4f}   (v12 base 0.70 / v14 large 0.7129 와 비교)")
    print(classification_report(y[va], pred, labels=list(range(len(ACTIONS))),
                                target_names=ACTIONS, digits=3, zero_division=0))
    np.save("holdout_probs3.npy", out.predictions)
    np.save("holdout_idx3.npy", va)
    if HOLDOUT_ONLY:
        with open("GATE_DONE", "w") as _gf:
            _gf.write(f"HOLDOUT_ONLY seed={SEED} holdout_f1={f1:.6f}\n")
        print(f"[HOLDOUT_ONLY] 밴드 자격만 계산 완료: holdout_probs3.npy/holdout_idx3.npy 저장, "
              f"홀드 Macro-F1={f1:.4f}. full-data refit/model_sub/script.py/requirements.txt/submit_base2.zip "
              f"생략, GATE_DONE 기록 후 종료.", flush=True)
        sys.exit(0)

# ---------- 전체 재학습(제출용) ----------
if SKIP_REFIT:
    print("[smoke] SKIP_REFIT=1 -> 홀드아웃 모델 저장")
    model_full, scaler_full = model, scaler
else:
    print("\nRefit on FULL data...")
    if not REFIT_ONLY:
        del trainer, model
        torch.cuda.empty_cache()
    _, model_full, scaler_full = train_model(np.arange(len(texts)))

if os.path.exists(SUB_DIR):
    shutil.rmtree(SUB_DIR)
os.makedirs(SUB_DIR)
model_full.backbone.save_pretrained(os.path.join(SUB_DIR, "backbone"))
tok.save_pretrained(os.path.join(SUB_DIR, "backbone"))
_tc = os.path.join(SUB_DIR, "backbone", "tokenizer_config.json")
# roberta/bert(WordPiece) 계열만 구버전 호환 패치. deberta/sentencepiece 계열은
# tokenizer_class를 건드리면 로딩 깨짐 → 백본명으로 분기.
_is_wordpiece = ("roberta" in MODEL_NAME.lower() or "bert" in MODEL_NAME.lower()) \
    and "deberta" not in MODEL_NAME.lower()
if os.path.exists(_tc) and _is_wordpiece:   # transformers 5.x 신형 포맷 -> 구버전 호환
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

# ---------- 제출용 script.py (단독 제출용 참고본 — v17 스태킹은 로컬 포장) ----------
SUBMIT_SCRIPT = r'''"""추론(제출용) — base2 하이브리드([SRC]+512) 단독."""
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
tok.truncation_side = "left"
samples = feat.load_jsonl("./data/test.jsonl")
texts = [build_transcript(s) for s in samples]
Mt = prep["scaler"].transform(
    feat.build_meta_frame(samples, columns=prep["meta_columns"]).values.astype(np.float32)
).astype(np.float32)

net = HybridNet("./model_sub/backbone", Mt.shape[1], len(ACTIONS))
net.head.load_state_dict(torch.load("./model_sub/head.pt", map_location="cpu"))
net = net.to(DEV).eval()
if DEV == "cuda":
    net = net.half()

order = np.argsort([len(t) for t in texts])
preds = np.zeros(len(texts), dtype=np.int64)
B = 128
with torch.no_grad():
    for i in range(0, len(texts), B):
        idx = order[i:i + B]
        b = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN,
                padding=True, pad_to_multiple_of=8, return_tensors="pt").to(DEV)
        m = torch.tensor(Mt[idx]).to(DEV)
        if DEV == "cuda":
            m = m.half()
        cls = net.backbone(**b).last_hidden_state[:, 0]
        preds[idx] = net.head(torch.cat([cls, m], -1)).float().argmax(-1).cpu().numpy()
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

zp = "submit_base2.zip"
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
_f1s = "N/A(REFIT_ONLY)" if REFIT_ONLY else f"{f1:.4f}"
print(f"\n[완료] {zp}  ({size:.0f} MB)  홀드아웃 Macro-F1={_f1s}")
