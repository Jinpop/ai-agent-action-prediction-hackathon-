#!/usr/bin/env python3
"""safetensors 변환본 전수 검증 (07-12 사용자 규약):
원본 pytorch_model.bin vs 변환 model.safetensors 를 CPU에서 전수 비교.
- state-dict key 집합 동일 / missing·unexpected 0
- 각 tensor shape·dtype 동일
- 모든 tensor torch.equal
- config·tokenizer 파일 SHA 대조 (캐시 원본 vs 로컬 dir)
통과 시에만 "가중치 바이트 동일"로 기록 가능.
"""
import glob, hashlib, os, sys, torch
import safetensors.torch as st

CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--kakaobank--kf-deberta-base/snapshots")
snap = sorted(glob.glob(CACHE + "/*/"))[0]
LOCAL = os.path.expanduser("~/dacon/kfdeberta_st")
fails = []
def chk(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok: fails.append(name)

binf = os.path.join(snap, "pytorch_model.bin")
stf = os.path.join(LOCAL, "model.safetensors")
chk("원본 bin 존재", os.path.exists(binf), binf)
chk("변환 safetensors 존재", os.path.exists(stf), stf)

sd_bin = torch.load(binf, map_location="cpu", weights_only=True)
sd_st = st.load_file(stf)
kb, ks = set(sd_bin), set(sd_st)
chk("key 집합 동일", kb == ks, f"missing={list(kb-ks)[:4]} unexpected={list(ks-kb)[:4]}")
chk("missing/unexpected 0", len(kb ^ ks) == 0, f"symdiff={len(kb^ks)}")

shape_bad, dtype_bad, neq, nonfinite = [], [], [], []
for k in sorted(kb & ks):
    a, b = sd_bin[k], sd_st[k]
    if tuple(a.shape) != tuple(b.shape): shape_bad.append(k)
    if a.dtype != b.dtype: dtype_bad.append((k, str(a.dtype), str(b.dtype)))
    if not torch.equal(a, b): neq.append(k)
    if a.is_floating_point() and not torch.isfinite(a).all(): nonfinite.append(k)
chk("shape 전수 동일", not shape_bad, str(shape_bad[:4]))
chk("dtype 전수 동일", not dtype_bad, str(dtype_bad[:4]))
chk("torch.equal 전수 통과", not neq, f"{len(neq)}개 불일치: {neq[:4]}")
chk("원본 finite", not nonfinite, str(nonfinite[:4]))
print(f"  텐서 {len(kb&ks)}개, 총 param {sum(v.numel() for v in sd_bin.values()):,}, dtype샘플={next(iter(sd_bin.values())).dtype}")

# config·tokenizer SHA 대조 (캐시 원본 vs 로컬 dir)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()
for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]:
    src, dst = os.path.join(snap, fn), os.path.join(LOCAL, fn)
    if os.path.exists(src) and os.path.exists(dst):
        chk(f"SHA {fn}", sha(src) == sha(dst))
    else:
        chk(f"{fn} 양쪽 존재", False, f"src={os.path.exists(src)} dst={os.path.exists(dst)}")

print()
print("VERDICT:", "ALL PASS — 가중치 바이트 동일 확정" if not fails else f"FAIL({len(fails)}): {fails}")
sys.exit(0 if not fails else 1)
