"""package-builder: submit_2nn_drop_{s48,k81,m2b79}.zip 결정론적 빌드.
campaign official-meta-wave1 Part A. 승인된 구성만 포장:
  cA(model_s48 + model_kf768_81 + model_kf768_m2b79 + classic)에서 seed 멤버 1개씩 제거한
  2NN+C ablation 3종. 전략/구성 변경 없음 — 지휘자 승인 근거에 따른 포장만.

절차(각 zip): fresh unzip(독립파일, cp -al 미사용) -> 제거멤버 dir 전체 rmtree ->
script.py의 SEED_DIRS 2멤버 편집 + docstring 교체(제거 멤버의 old dirname 리터럴 미포함) ->
결정론적 rezip(고정 타임스탬프, 금지엔트리 배제).
"""
import hashlib
import os
import shutil
import zipfile

BASE = "/Users/<USER>/Documents/Dacon_236694_AI_Agent"
SRC_ZIP = os.path.join(BASE, "submits/submit_cA.zip")
FIXED_DT = (2026, 7, 14, 0, 0, 0)  # 결정론적 재현 (build_m2v2_combo.py와 동일 컨벤션)

DIR_ATTR = (0o40755 << 16) | 0x10
FILE_ATTR = (0o100644 << 16)

FORBIDDEN_NAMES = {"__MACOSX", ".DS_Store", "__pycache__"}

NEURAL_ITEMS = [
    "head.pt", "prep.pkl",
    ("backbone", [
        "model.safetensors", "tokenizer_config.json", "special_tokens_map.json",
        "config.json", "tokenizer.json", "vocab.txt",
    ]),
]
CLASSIC_ITEMS = ["artifacts.pkl"]

# cA 내부 원 순서 (member_order 골격, build_m2v2_combo.py의 cA 순서와 동일)
FULL_ORDER = ["model_s48", "requirements.txt", "model_kf768_81", "model_kf768_m2b79", "model"]

MEMBER_ITEMS = {
    "model_s48": NEURAL_ITEMS,
    "model_kf768_81": NEURAL_ITEMS,
    "model_kf768_m2b79": NEURAL_ITEMS,
    "model": CLASSIC_ITEMS,
}

DOCSTRINGS = {
    "model_s48": '''"""추론 (official-meta-wave1 Part A ablation) — 2NN+C. 챔피언 cA 3-seed 구성에서 seed 1개 제거,
잔존 model_kf768_81 + model_kf768_m2b79 + 고전 3모델 블렌드.

결합기 없음(순수 확률 블렌드, BLEND_W=0.6) — "홀드아웃에 덜 맞출수록 서버 보너스가 크다"는
검증된 법칙에 따름.

구성:
  ./model/artifacts.pkl              고전 HGB/LogReg/NB (블렌드 0.45/0.40/0.15)
  ./model_kf768_81 / ./model_kf768_m2b79   (전원 fp16)

시간 예산: 고전 ~60s + seed당 ~100s ≈ 4분(2seed). 데드라인 초과/오류 seed는 평균에서 제외,
seed가 하나도 없으면 고전 블렌드만으로 예측 (무결합기라 자연스러운 강등).
반드시 net.half() (from_pretrained 는 fp32 업캐스트).
"""''',
    "model_kf768_81": '''"""추론 (official-meta-wave1 Part A ablation) — 2NN+C. 챔피언 cA 3-seed 구성에서 seed 1개 제거,
잔존 model_s48 + model_kf768_m2b79 + 고전 3모델 블렌드.

결합기 없음(순수 확률 블렌드, BLEND_W=0.6) — "홀드아웃에 덜 맞출수록 서버 보너스가 크다"는
검증된 법칙에 따름.

구성:
  ./model/artifacts.pkl              고전 HGB/LogReg/NB (블렌드 0.45/0.40/0.15)
  ./model_s48 / ./model_kf768_m2b79   (전원 fp16)

시간 예산: 고전 ~60s + seed당 ~100s ≈ 4분(2seed). 데드라인 초과/오류 seed는 평균에서 제외,
seed가 하나도 없으면 고전 블렌드만으로 예측 (무결합기라 자연스러운 강등).
반드시 net.half() (from_pretrained 는 fp32 업캐스트).
"""''',
    "model_kf768_m2b79": '''"""추론 (official-meta-wave1 Part A ablation) — 2NN+C. 챔피언 cA 3-seed 구성에서 seed 1개 제거,
잔존 model_s48 + model_kf768_81 + 고전 3모델 블렌드.

결합기 없음(순수 확률 블렌드, BLEND_W=0.6) — "홀드아웃에 덜 맞출수록 서버 보너스가 크다"는
검증된 법칙에 따름.

구성:
  ./model/artifacts.pkl              고전 HGB/LogReg/NB (블렌드 0.45/0.40/0.15)
  ./model_s48 / ./model_kf768_81   (전원 fp16)

시간 예산: 고전 ~60s + seed당 ~100s ≈ 4분(2seed). 데드라인 초과/오류 seed는 평균에서 제외,
seed가 하나도 없으면 고전 블렌드만으로 예측 (무결합기라 자연스러운 강등).
반드시 net.half() (from_pretrained 는 fp32 업캐스트).
"""''',
}

CONFIGS = [
    {
        "removed": "model_s48",
        "seed_dirs": ["./model_kf768_81", "./model_kf768_m2b79"],
        "staging": os.path.join(BASE, "scratchpad/stage_2nn_drop_s48"),
        "out_zip": os.path.join(BASE, "submits/submit_2nn_drop_s48.zip"),
    },
    {
        "removed": "model_kf768_81",
        "seed_dirs": ["./model_s48", "./model_kf768_m2b79"],
        "staging": os.path.join(BASE, "scratchpad/stage_2nn_drop_k81"),
        "out_zip": os.path.join(BASE, "submits/submit_2nn_drop_k81.zip"),
    },
    {
        "removed": "model_kf768_m2b79",
        "seed_dirs": ["./model_s48", "./model_kf768_81"],
        "staging": os.path.join(BASE, "scratchpad/stage_2nn_drop_m2b79"),
        "out_zip": os.path.join(BASE, "submits/submit_2nn_drop_m2b79.zip"),
    },
]


def add_dir(zf, arcname):
    zi = zipfile.ZipInfo(arcname + "/", date_time=FIXED_DT)
    zi.external_attr = DIR_ATTR
    zi.compress_type = zipfile.ZIP_STORED
    zf.writestr(zi, b"")


def add_file(zf, real_path, arcname):
    zi = zipfile.ZipInfo(arcname, date_time=FIXED_DT)
    zi.external_attr = FILE_ATTR
    zi.compress_type = zipfile.ZIP_DEFLATED
    with open(real_path, "rb") as f:
        data = f.read()
    zf.writestr(zi, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def edit_script(staging, removed, new_seed_dirs):
    """staging/script.py(fresh unzip 독립파일)를 직접 in-place 편집.
    docstring 교체(제거 멤버 old dirname 리터럴 미포함) + SEED_DIRS 리스트 교체.
    SEED_DEADLINES는 그대로 둠(zip() 자동 truncate로 2멤버만 페어링, 문제없음).
    """
    path = os.path.join(staging, "script.py")
    text = open(path, encoding="utf-8").read()

    # 1) docstring 교체: 파일 맨 앞 첫 번째 트리플쿼트 블록
    assert text.startswith('"""'), "script.py는 트리플쿼트 docstring으로 시작해야 함"
    end = text.index('"""', 3) + 3
    old_docstring = text[:end]
    new_docstring = DOCSTRINGS[removed]
    text = new_docstring + text[end:]

    # 2) SEED_DIRS 라인 교체 (원본 3-seed 리스트 리터럴을 정확히 대체)
    old_line = "SEED_DIRS = ['./model_s48', './model_kf768_81', './model_kf768_m2b79']"
    assert old_line in text, "원본 SEED_DIRS 리터럴을 찾지 못함"
    new_line = "SEED_DIRS = [" + ", ".join(f"'{d}'" for d in new_seed_dirs) + "]"
    text = text.replace(old_line, new_line)

    # 안전검사: 제거 멤버의 basename 리터럴이 파일 어디에도 남지 않았는지 확인
    assert removed not in text, f"제거 멤버 리터럴 잔존: {removed}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return old_docstring, new_docstring


def scan_forbidden(staging):
    hits = []
    for root, dirs, files in os.walk(staging):
        bad = FORBIDDEN_NAMES & (set(dirs) | set(files))
        if bad:
            hits.append((root, bad))
        for f in files:
            lf = f.lower()
            if lf.endswith(".pyc") or "checkpoint" in lf or "memo" in lf:
                hits.append((root, {f}))
    if os.path.exists(os.path.join(staging, "data")) or os.path.exists(os.path.join(staging, "output")):
        hits.append((staging, {"data/ or output/"}))
    return hits


def build_one(cfg):
    removed = cfg["removed"]
    staging = cfg["staging"]
    out_zip = cfg["out_zip"]

    # 0) fresh staging (독립파일; cp -al 미사용 — extractall은 항상 새 파일 데이터를 씀)
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)
    with zipfile.ZipFile(SRC_ZIP) as zf:
        zf.extractall(staging)

    # 1) 제거 멤버 dir 전체 삭제
    removed_path = os.path.join(staging, removed)
    assert os.path.isdir(removed_path), f"제거 대상 dir 없음: {removed_path}"
    shutil.rmtree(removed_path)

    # 2) script.py 편집 (fresh 파일 직접 in-place — 하드링크 아님)
    edit_script(staging, removed, cfg["seed_dirs"])

    # 3) 금지 엔트리 사전 스캔
    hits = scan_forbidden(staging)
    assert not hits, f"금지 엔트리 존재: {hits}"

    # 4) 결정론적 rezip (member_order에서 제거 멤버만 skip, 나머지 원 순서 유지)
    order = [x for x in FULL_ORDER if x != removed]
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    if os.path.exists(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w") as zf:
        for entry in order:
            if entry == "requirements.txt":
                add_file(zf, os.path.join(staging, "requirements.txt"), "requirements.txt")
                continue
            items = MEMBER_ITEMS[entry]
            add_dir(zf, entry)
            for item in items:
                if isinstance(item, tuple):
                    subdir, subitems = item
                    add_dir(zf, f"{entry}/{subdir}")
                    for fn in subitems:
                        add_file(zf, os.path.join(staging, entry, subdir, fn), f"{entry}/{subdir}/{fn}")
                else:
                    add_file(zf, os.path.join(staging, entry, item), f"{entry}/{item}")
        add_file(zf, os.path.join(staging, "script.py"), "script.py")
        add_file(zf, os.path.join(staging, "feat.py"), "feat.py")

    size = os.path.getsize(out_zip)
    sha = sha256_file(out_zip)
    return {
        "zip": out_zip,
        "size_bytes": size,
        "size_ok": size < 1_000_000_000,
        "sha256": sha,
        "staging_dir": staging,
        "removed_member": removed,
        "new_seed_dirs": cfg["seed_dirs"],
    }


if __name__ == "__main__":
    results = []
    for cfg in CONFIGS:
        r = build_one(cfg)
        results.append(r)
        print(r)
    print()
    for r in results:
        print(f"{r['zip']}  size={r['size_bytes']:,}  ok={r['size_ok']}  sha256={r['sha256']}")
