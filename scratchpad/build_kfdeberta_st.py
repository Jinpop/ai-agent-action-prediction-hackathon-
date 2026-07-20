#!/usr/bin/env python3
"""kf-deberta-base pristine safetensors 번들 구성 (Elice torch<2.6 CVE-2025-32434 우회용).
Mac torch 2.8(>=2.6)로 HF캐시 .bin을 from_pretrained → save_pretrained(safe_serialization=True).
config+tokenizer+model.safetensors 포함 로컬 디렉터리 생성. base canary(params/vocab) 검증.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import hashlib
from transformers import AutoModel, AutoTokenizer, AutoConfig
NAME = "kakaobank/kf-deberta-base"
OUT = "/Users/<USER>/Documents/Dacon_236694_AI_Agent/scratchpad/kfdeberta_st"

cfg = AutoConfig.from_pretrained(NAME)
print(f"[cfg] model_type={cfg.model_type} hidden={cfg.hidden_size} vocab={cfg.vocab_size} layers={cfg.num_hidden_layers}")
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME)  # DebertaV2Model (base encoder)
n_params = sum(p.numel() for p in model.parameters())
print(f"[canary] params={n_params} (기대 185,290,752)  vocab={cfg.vocab_size} (기대 130,000)  tok_vocab={tok.vocab_size}")
assert n_params == 185290752, f"params 불일치 {n_params}"
assert cfg.vocab_size == 130000, f"vocab 불일치 {cfg.vocab_size}"

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print(f"[write] {OUT}")
print("[files]", sorted(os.listdir(OUT)))
st = os.path.join(OUT, "model.safetensors")
assert os.path.exists(st), "model.safetensors 미생성 (shared-tensor 문제?)"
assert os.path.exists(os.path.join(OUT, "config.json")), "config.json 없음"
h = hashlib.sha256(open(st, "rb").read()).hexdigest()
print(f"[sha] model.safetensors {os.path.getsize(st)}B  sha256={h}")

# 재로드 검증(무결성): 저장된 safetensors 디렉터리에서 다시 로드 → params 동일
m2 = AutoModel.from_pretrained(OUT)
n2 = sum(p.numel() for p in m2.parameters())
assert n2 == n_params, f"재로드 params 불일치 {n2}"
print(f"[verify] safetensors 재로드 params={n2} ✓ (from_pretrained 정상)")
