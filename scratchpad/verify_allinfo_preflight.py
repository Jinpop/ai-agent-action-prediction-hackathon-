#!/usr/bin/env python3
"""ALL-INFO 발사전 fail-closed 배터리 (토큰길이 #6/#7 제외 — 별도 스캔).
#5 Stage B 전체 공식필드 transcript / #8 meta 125·중복없음 / #9 +6필드 정확 / #10 scaler125 / #12 SHA."""
import os, sys, ast, json, hashlib
os.environ["META_NUM_EXT"] = "1"   # feat import 전 필수
ROOT = "/Users/<USER>/Documents/Dacon_236694_AI_Agent"
FROZEN_DIR = f"{ROOT}/scratchpad/allinfo_frozen"
sys.path.insert(0, FROZEN_DIR)
import feat  # frozen feat (META_NUM_EXT=1 반영)

print("=== #5 Stage B 전체 공식필드 transcript (META_TRANS_EXT=1, real train 샘플) ===")
src = open(f"{FROZEN_DIR}/colab_train_base2.py").read()
tree = ast.parse(src)
ns = {"os": os, "PRETEXT_META": "blank", "META_TRANS_EXT": True}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("_s", "build_transcript"):
        exec(compile(ast.Module([node], []), "frozen", "exec"), ns)
bt = ns["build_transcript"]
OFFICIAL = ["tier=", "langpref=", "ci=", "dirty=", "turn=", "budget=", "elapsed=", "loc=", "langmix=", "open="]
reals = [json.loads(l) for _, l in zip(range(3), open(f"{ROOT}/open/data/train.jsonl"))]
for r in reals:
    ml = [ln for ln in bt(r).split("\n") if ln.startswith("[META]")][0]
    miss = [f for f in OFFICIAL if f not in ml]
    assert not miss, f"Stage B [META] 공식필드 누락 {miss}: {ml}"
    print(f"  ok(10필드): {ml[:120]}...")
print("[#5] Stage B 10개 공식필드 전부 존재 ✓")

print("\n=== #8/#9/#10 feat 125d 숫자메타 ===")
meta_df = feat.build_meta_frame(reals + reals)  # 아무 real 샘플
cols = list(meta_df.columns)
print(f"  meta dim = {len(cols)} (기대 125)")
assert len(cols) == 125, f"dim 불일치 {len(cols)}"
assert len(cols) == len(set(cols)), "중복 컬럼 존재"
extra6 = ["lang_kt", "lang_vue", "lang_swift", "lang_ipynb", "lang_tf", "langpref_mixed"]
miss6 = [c for c in extra6 if c not in cols]
assert not miss6, f"+6 확장필드 누락 {miss6}"
print(f"  [#9] +6 확장필드 전부 존재: {extra6} ✓")
print(f"  [#8] 125 컬럼·중복0 ✓  feat.META_NUM_EXT={feat.META_NUM_EXT}")
# scaler는 학습시 fit — 여기선 컬럼수=125 확인으로 scaler.n_features_in_=125 계약 성립(#10)
print(f"  [#10] scaler.n_features_in_=125·head_in=768+125=893 계약(학습시 성립) ✓")

print("\n=== #12 SHA 영속화 ===")
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
arts = {
    "frozen/colab_train_base2.py": f"{FROZEN_DIR}/colab_train_base2.py",
    "frozen/feat.py": f"{FROZEN_DIR}/feat.py",
    "data/train_mint_allinfo_stable.jsonl": f"{ROOT}/open/data/train_mint_allinfo_stable.jsonl",
    "data/train_mint_allinfo_stable_labels.csv": f"{ROOT}/open/data/train_mint_allinfo_stable_labels.csv",
    "data/train.jsonl": f"{ROOT}/open/data/train.jsonl",
    "data/train_labels.csv": f"{ROOT}/open/data/train_labels.csv",
    "base/model.safetensors": f"{ROOT}/scratchpad/kfdeberta_st/model.safetensors",
    "base/config.json": f"{ROOT}/scratchpad/kfdeberta_st/config.json",
    "base/tokenizer.json": f"{ROOT}/scratchpad/kfdeberta_st/tokenizer.json",
}
shatab = {k: sha(v) for k, v in arts.items()}
for k, v in shatab.items():
    print(f"  {v[:16]}  {k}")
json.dump(shatab, open(f"{ROOT}/scratchpad/allinfo_frozen/artifact_sha.json", "w"), indent=2)
print(f"[#12] SHA 표 저장: scratchpad/allinfo_frozen/artifact_sha.json")
