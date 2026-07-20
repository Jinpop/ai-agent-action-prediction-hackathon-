"""package-builder: submit_m2v2s{79,81}.zip 결정론적 빌드.
승인된 구성(mint2v2-dualseed-0714 campaign, cA에서 model_kf768_m2b79 -> model_kf768_m2v2sX 슬롯교체)만
staging에서 zip으로 포장한다. 전략/구성 변경 없음.
"""
import os
import stat
import zipfile

BASE = "/Users/<USER>/Documents/Dacon_236694_AI_Agent"
FIXED_DT = (2026, 7, 14, 0, 0, 0)  # 결정론적 재현을 위한 고정 타임스탬프

DIR_ATTR = (0o40755 << 16) | 0x10   # directory, unix perm 755 + MS-DOS dir bit
FILE_ATTR = (0o100644 << 16)        # regular file, unix perm 644

FORBIDDEN_NAMES = {"__MACOSX", ".DS_Store", "__pycache__"}


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


def build(staging, new_member_dir, out_zip):
    # sanity: no forbidden entries anywhere in staging
    for root, dirs, files in os.walk(staging):
        bad = FORBIDDEN_NAMES & (set(dirs) | set(files))
        if bad:
            raise RuntimeError(f"forbidden entries present in staging: {bad} under {root}")
        low = {f.lower() for f in files}
        for f in files:
            if f.lower().endswith(".pyc") or "checkpoint" in f.lower() or "memo" in f.lower():
                raise RuntimeError(f"forbidden file present: {os.path.join(root, f)}")
    if os.path.exists(os.path.join(staging, "data")) or os.path.exists(os.path.join(staging, "output")):
        raise RuntimeError("forbidden data/ or output/ dir present in staging")

    member_order = [
        ("model_s48", [
            "head.pt", "prep.pkl",
            ("backbone", [
                "model.safetensors", "tokenizer_config.json", "special_tokens_map.json",
                "config.json", "tokenizer.json", "vocab.txt",
            ]),
        ]),
        None,  # placeholder for requirements.txt (top-level, mid-sequence like cA)
        ("model_kf768_81", [
            "head.pt", "prep.pkl",
            ("backbone", [
                "model.safetensors", "tokenizer_config.json", "special_tokens_map.json",
                "config.json", "tokenizer.json", "vocab.txt",
            ]),
        ]),
        (new_member_dir, [
            "head.pt", "prep.pkl",
            ("backbone", [
                "model.safetensors", "tokenizer_config.json", "special_tokens_map.json",
                "config.json", "tokenizer.json", "vocab.txt",
            ]),
        ]),
        ("model", ["artifacts.pkl"]),
    ]

    out_path = os.path.join(BASE, out_zip)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    with zipfile.ZipFile(out_path, "w") as zf:
        for entry in member_order:
            if entry is None:
                add_file(zf, os.path.join(staging, "requirements.txt"), "requirements.txt")
                continue
            dirname, items = entry
            add_dir(zf, dirname)
            for item in items:
                if isinstance(item, tuple):
                    subdir, subitems = item
                    add_dir(zf, f"{dirname}/{subdir}")
                    for fn in subitems:
                        add_file(zf, os.path.join(staging, dirname, subdir, fn), f"{dirname}/{subdir}/{fn}")
                else:
                    add_file(zf, os.path.join(staging, dirname, item), f"{dirname}/{item}")
        add_file(zf, os.path.join(staging, "script.py"), "script.py")
        add_file(zf, os.path.join(staging, "feat.py"), "feat.py")

    size = os.path.getsize(out_path)
    return out_path, size


if __name__ == "__main__":
    for X in ["79", "81"]:
        staging = f"{BASE}/scratchpad/stage_m2v2s{X}"
        new_member = f"model_kf768_m2v2s{X}"
        out_zip = f"submits/submit_m2v2s{X}.zip"
        path, size = build(staging, new_member, out_zip)
        print(X, path, size, "bytes", "OK<1e9" if size < 1_000_000_000 else "FAIL>=1e9")
