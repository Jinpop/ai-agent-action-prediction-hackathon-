#!/usr/bin/env python3
"""HF 캐시의 kf-deberta-base pytorch_model.bin → model.safetensors 변환.
목적: transformers 4.57.6 + torch 2.5.1 조합의 torch.load 차단(CVE-2025-32434) 우회.
가중치 바이트 동일(safetensors는 컨테이너만 상이) → 학습 드로우 무영향. 스크립트 무수정.
"""
import glob, os, sys, torch
import safetensors.torch as st

REPO = sys.argv[1] if len(sys.argv) > 1 else "models--kakaobank--kf-deberta-base"
hub = os.path.expanduser(f"~/.cache/huggingface/hub/{REPO}/snapshots")
snaps = sorted(glob.glob(hub + "/*/"))
assert snaps, f"snapshot 없음: {hub}"
for snap in snaps:
    binf = os.path.join(snap, "pytorch_model.bin")
    stf = os.path.join(snap, "model.safetensors")
    if os.path.exists(stf) and not os.path.islink(stf):
        print(f"[skip] 이미 존재: {stf}")
        continue
    if not os.path.exists(binf):
        print(f"[skip] bin 없음: {binf}")
        continue
    sd = torch.load(binf, map_location="cpu", weights_only=True)
    # 텐서만, contiguous clone (공유 스토리지 방지)
    clean = {k: v.clone().contiguous() for k, v in sd.items() if hasattr(v, "clone")}
    st.save_file(clean, stf, metadata={"format": "pt"})
    print(f"[ok] {stf}  ({len(clean)} tensors, {round(os.path.getsize(stf)/1e6)} MB)")
print("DONE")
