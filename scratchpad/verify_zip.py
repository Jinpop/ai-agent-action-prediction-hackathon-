#!/usr/bin/env python3
"""제출 zip 강제 검증 (07-12 규약 §zip자체 + §zip생성전 구조 항목).
Usage: verify_zip.py <zip> <staging_dir> <ref_staging(직전 프레임: v49 또는 v58)> <swapped_new_dir> <swapped_old_dir> [member_type]
  member_type: kf768|s48|dv3ko (미지정시 NEWM dirname으로 자동탐지). MEMBER_SPECS 참조.
검사: 크기<1e9 / CRC / 경로위생 / 중복경로 / 금지파일 / zip↔staging CRC32 전수 /
     공통파일 REF 바이트동일(diff는 script.py+교체멤버만) / 교체멤버 비복사본(SHA) /
     backbone fp16·finite / head finite·(hidden+119)→256→14 / prep actions==feat.ACTIONS /
     max_len·params(member-spec) / SEED_DIRS 스왑 정합 / backbone config(model_type·hidden·vocab)
★07-13 일반화: v49/kf768 전용 → member-spec 레지스트리로 kf768·s48·dv3ko 지원.
  단 T4 600초 런타임·SPM 토크나이저 라운드트립은 여전히 verify_zip 범위 밖(unverified 명시·5행 E2E/서버 probe로).
"""
import sys, os, zipfile, zlib, hashlib, json
sys.path.insert(0, "open/scripts")

ZIP, STG, REF, NEWM, OLDM = sys.argv[1:6]

# ★07-13 member-spec 레지스트리 (dirname 추측 제거). 신규 백본은 여기에 spec 추가.
MEMBER_SPECS = {
    "kf768": {"max_len": 768, "params": 185_290_752, "hidden": 768, "vocab": 130000, "model_type": "deberta-v2"},
    "s48":   {"max_len": 512, "params": 110_618_112, "hidden": 768, "vocab": None,   "model_type": "roberta"},
    "dv3ko": {"max_len": 768, "params": 134_679_552, "hidden": 768, "vocab": 64100,  "model_type": "deberta-v2"},
}
def _detect_type(dirname):
    b = os.path.basename(dirname.rstrip("/")).lower()
    if "dv3ko" in b or "dvko" in b: return "dv3ko"
    if "kf768" in b: return "kf768"
    if "s48" in b or b.startswith("model_s"): return "s48"
    return None
MTYPE = sys.argv[6] if len(sys.argv) > 6 else _detect_type(NEWM)
assert MTYPE in MEMBER_SPECS, f"member_type 미지정/미탐지 (NEWM={NEWM}); 6번째 인자로 {list(MEMBER_SPECS)} 중 지정"
SPEC = MEMBER_SPECS[MTYPE]
print(f"[member_type] {MTYPE} spec={SPEC}")
fails = []
def chk(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok: fails.append(name)

# 1) 크기
sz = os.path.getsize(ZIP)
chk("size<1e9", sz < 1_000_000_000, f"{sz:,}")

zf = zipfile.ZipFile(ZIP)
# 2) CRC 무결성
chk("zip CRC(testzip)", zf.testzip() is None)
names = zf.namelist()
files = [n for n in names if not n.endswith("/")]
# 3) 경로 위생
chk("no absolute/..", all(not n.startswith("/") and ".." not in n for n in names))
chk("no dup paths", len(set(names)) == len(names))
BAD = ("__MACOSX", ".DS_Store", "checkpoint", "memo", "__pycache__", ".pyc")
bad_hits = [n for n in names if any(b in n for b in BAD) or n.startswith(("data/", "output/"))]
chk("no forbidden entries", not bad_hits, str(bad_hits[:5]))

# 4) zip ↔ staging CRC32 전수
def crc32_file(p):
    c = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            c = zlib.crc32(chunk, c)
    return c & 0xFFFFFFFF
mism, missing = [], []
for zi in zf.infolist():
    if zi.filename.endswith("/"): continue
    p = os.path.join(STG, zi.filename)
    if not os.path.exists(p): missing.append(zi.filename); continue
    if crc32_file(p) != zi.CRC: mism.append(zi.filename)
stg_files = set()
for root, _, fs in os.walk(STG):
    for f in fs:
        stg_files.add(os.path.relpath(os.path.join(root, f), STG))
extra = stg_files - set(files)
chk("zip↔staging CRC32 전수일치", not mism and not missing and not extra,
    f"mismatch={mism[:3]} missing={missing[:3]} staging-only={list(extra)[:3]}")

# 5) v49 대비: 공통파일 바이트 동일, diff는 script.py + 교체멤버만
diffs = []
ref_files = set()
for root, _, fs in os.walk(REF):
    for f in fs:
        ref_files.add(os.path.relpath(os.path.join(root, f), REF))
for rel in sorted(stg_files & ref_files):
    a, b = os.path.join(STG, rel), os.path.join(REF, rel)
    if os.path.getsize(a) != os.path.getsize(b) or crc32_file(a) != crc32_file(b):
        diffs.append(rel)
only_new = sorted(stg_files - ref_files)
only_ref = sorted(ref_files - stg_files)
chk("공통파일 diff == {script.py}", diffs == ["script.py"], str(diffs[:5]))
chk("신규파일 == 교체멤버뿐", all(p.startswith(NEWM.rstrip('/') + "/") for p in only_new), str(only_new[:5]))
chk("제거파일 == 구멤버뿐", all(p.startswith(OLDM.rstrip('/') + "/") for p in only_ref), str(only_ref[:5]))

# 6) 교체멤버 비복사본: head/backbone SHA가 v49 모든 멤버와 상이
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
new_head = sha(os.path.join(STG, NEWM, "head.pt"))
new_bb = sha(os.path.join(STG, NEWM, "backbone", "model.safetensors"))
dup = []
for d in os.listdir(REF):
    hp = os.path.join(REF, d, "head.pt")
    if os.path.isfile(hp):
        if sha(hp) == new_head: dup.append(d + "/head.pt")
        bp = os.path.join(REF, d, "backbone", "model.safetensors")
        if os.path.isfile(bp) and sha(bp) == new_bb: dup.append(d + "/backbone")
chk("교체멤버 비복사본(SHA 상이)", not dup, str(dup))
print(f"  신규멤버 SHA: head={new_head[:16]}… backbone={new_bb[:16]}…")

# 7) 모델 계약
import torch, numpy as np, joblib
from safetensors import safe_open
import feat
mdir = os.path.join(STG, NEWM)
prep = joblib.load(os.path.join(mdir, "prep.pkl"))
chk("prep[actions]==feat.ACTIONS", list(prep.get("actions", [])) == list(feat.ACTIONS))
max_len = int(prep.get("max_len", -1))
chk("max_len 계약", max_len == SPEC["max_len"], f"max_len={max_len} (type={MTYPE})")
tot, bad_dtype, nonfinite = 0, [], False
with safe_open(os.path.join(mdir, "backbone", "model.safetensors"), framework="pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        tot += t.numel()
        if t.dtype != torch.float16: bad_dtype.append((k, str(t.dtype)))
        if t.is_floating_point() and not torch.isfinite(t).all(): nonfinite = True
chk("backbone fp16 전체", not bad_dtype, str(bad_dtype[:3]))
chk("backbone finite", not nonfinite)
chk(f"params 계약({MTYPE})", tot == SPEC["params"], f"{tot:,} (기대 {SPEC['params']:,})")
sd = torch.load(os.path.join(mdir, "head.pt"), map_location="cpu")
w0 = sd.get("net.0.weight") if sd.get("net.0.weight") is not None else sd.get("0.weight")
w3 = sd.get("net.3.weight") if sd.get("net.3.weight") is not None else sd.get("3.weight")
# ★meta 계약 강화 (campaign official-meta-wave1 #1): len(meta_columns)만 신뢰하지 말 것
_meta_cols = list(prep.get("meta_columns") or []) if isinstance(prep, dict) else []
_meta_dim = len(_meta_cols) if _meta_cols else 119
chk("meta_columns 중복 없음", len(_meta_cols) == len(set(_meta_cols)), f"n={_meta_dim}")
_scn = getattr(prep.get("scaler"), "n_features_in_", None) if isinstance(prep, dict) else None
chk("scaler.n_features_in_ == len(meta_columns)", _scn is None or _scn == _meta_dim, f"scaler={_scn} cols={_meta_dim}")
if _meta_dim != 119:   # META-N(125): 기존 119 대비 정확히 +6, 제거 0
    _META_N_EXTRA = {"lang_kt", "lang_vue", "lang_swift", "lang_ipynb", "lang_tf", "langpref_mixed"}
    _op = os.path.join(REF, OLDM, "prep.pkl")
    _ref_cols = set(joblib.load(_op).get("meta_columns") or []) if os.path.isfile(_op) else set()
    _added, _removed = set(_meta_cols) - _ref_cols, _ref_cols - set(_meta_cols)
    chk("meta +6 정확(vs ref 119)", _meta_dim == 125 and _added == _META_N_EXTRA and not _removed,
        f"dim={_meta_dim} added={sorted(_added)} removed={sorted(_removed)}")
exp_in = SPEC["hidden"] + _meta_dim   # HybridNet: backbone CLS(hidden) ⊕ meta(_meta_dim)d — META-N=125→893, 기존=119→887
if w0 is None or w3 is None:  # 최후: 2D 텐서에서 입력 exp_in / 출력 14 탐색
    two_d = [v for v in sd.values() if getattr(v, "ndim", 0) == 2]
    w0 = next((v for v in two_d if v.shape[1] == exp_in), None)
    w3 = next((v for v in two_d if v.shape[0] == 14), None)
chk(f"head {exp_in}→256→14", w0 is not None and tuple(w0.shape) == (256, exp_in) and tuple(w3.shape) == (14, 256),
    f"{tuple(w0.shape) if w0 is not None else list(sd.keys())[:4]}→{tuple(w3.shape) if w3 is not None else None}")
chk("head finite", all(torch.isfinite(v).all() for v in sd.values()))

# 7b) backbone config 정합 (model_type·hidden·vocab) — dv3ko 등 백본 손상 조기 검출
cfgp = os.path.join(mdir, "backbone", "config.json")
if os.path.isfile(cfgp):
    bcfg = json.load(open(cfgp))
    chk(f"config model_type=={SPEC['model_type']}", bcfg.get("model_type") == SPEC["model_type"], str(bcfg.get("model_type")))
    chk(f"config hidden_size=={SPEC['hidden']}", bcfg.get("hidden_size") == SPEC["hidden"], str(bcfg.get("hidden_size")))
    if SPEC["vocab"] is not None:
        chk(f"config vocab_size=={SPEC['vocab']}", bcfg.get("vocab_size") == SPEC["vocab"], str(bcfg.get("vocab_size")))
else:
    chk("backbone config.json 존재", False, cfgp)

# 8) script 실행 계약
st = open(os.path.join(STG, "script.py"), encoding="utf-8").read()
chk("truncation_side=left", 'truncation_side = "left"' in st or "truncation_side='left'" in st or 'truncation_side="left"' in st)
chk("net.half() on cuda", "net.half()" in st)
chk("BLEND_W=0.6", "BLEND_W = 0.6" in st)
chk("classic 0.45/0.40/0.15", "0.45, 0.40, 0.15" in st.replace("(", "").replace(")", ""))
# ★SEED_DIRS 스왑 정합 (모든 스왑 공통 latent gap: 파일만 바꾸고 script 편집 누락 → 무증상 폴백)
_nb, _ob = os.path.basename(NEWM.rstrip("/")), os.path.basename(OLDM.rstrip("/"))
chk("SEED_DIRS 신규멤버 포함", _nb in st, _nb)
chk("SEED_DIRS 구멤버 제거됨", _ob not in st, _ob)

print()
print("VERDICT:", "ALL PASS" if not fails else f"FAIL({len(fails)}): {fails}")
sys.exit(0 if not fails else 1)
