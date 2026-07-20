#!/usr/bin/env python3
"""fallback zip 빌더: v49 프레임(s48+k79+k78)에서 k78 슬롯을 대체 kf 멤버로 교체.
결과 = s48 + kf768_79 + <새멤버>. 스크립트 무수정(SEED_DIRS 1줄 + docstring만).
usage: build_fallback.py <tag> <pack_dir> <member_dirname> <새멤버설명>
"""
import os, shutil, subprocess, sys
BASE = "/Users/<USER>/Documents/Dacon_236694_AI_Agent"
tag, pack, member, desc = sys.argv[1:5]
S = f"{BASE}/scratchpad"
stage = f"{S}/stage_{tag}"
shutil.rmtree(stage, ignore_errors=True)
subprocess.run(["cp", "-a", f"{S}/v49", stage], check=True)
shutil.rmtree(f"{stage}/model_kf768_78", ignore_errors=True)
os.makedirs(f"{stage}/{member}", exist_ok=True)
src = f"{S}/{pack}/model_sub"
shutil.copy2(f"{src}/head.pt", f"{stage}/{member}/head.pt")
shutil.copy2(f"{src}/prep.pkl", f"{stage}/{member}/prep.pkl")
subprocess.run(["cp", "-a", f"{src}/backbone", f"{stage}/{member}/backbone"], check=True)
# script.py 수정
p = f"{stage}/script.py"
t = open(p, encoding="utf-8").read()
subs = [
 ("'./model_kf768_78'", f"'./{member}'"),
 ('"""추론 v49 (제출용) — 챔피언 고전 1 + s48 + kf768_seed79 + kf768_seed78.',
  f'"""추론 {tag} (제출용) — 챔피언 고전 1 + s48 + kf768_seed79 + {desc}.'),
 ('v47(LB 0.7820)에서 kf768_74 → kf768_79 단일 교체(깨끗한 seed-only 대조).',
  f'v49(LB 0.7832)에서 kf768_78 → {desc} 단일 교체(fallback 후보).'),
 ('순수 확률 블렌드 (w-probe: 0.5)', '순수 확률 블렌드 (BLEND_W=0.6)'),
 ('  ./model_s45 / _s46 / _s47          base [SRC]+512 (fp16; 6ep/7ep/8ep, seed 45/46/47)',
  f'  ./model_s48 / model_kf768_79 / {member}   (전원 fp16)'),
]
for old, new in subs:
    assert t.count(old) == 1, (tag, old[:40], t.count(old))
    t = t.replace(old, new)
open(p, "w", encoding="utf-8").write(t)
# zip (pycache 제외)
shutil.rmtree(f"{stage}/__pycache__", ignore_errors=True)
zpath = f"{BASE}/submits/submit_{tag}.zip"
if os.path.exists(zpath): os.remove(zpath)
subprocess.run(["zip", "-rq", zpath, ".", "-x", "__pycache__/*"], cwd=stage, check=True)
sz = os.path.getsize(zpath)
print(f"{tag}: {sz:,} bytes under1GB={sz<10**9} member={member} SEED_DIRS={[l for l in t.splitlines() if 'SEED_DIRS =' in l]}")
