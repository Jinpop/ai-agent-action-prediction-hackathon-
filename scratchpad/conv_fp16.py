import os, sys
from transformers import AutoModel
r = sys.argv[1]
p = f"pack_{r}/model_sub/backbone"
m = AutoModel.from_pretrained(p).half()
m.save_pretrained(p, safe_serialization=True)
print(f"{r} fp16:", round(os.path.getsize(p+"/model.safetensors")/1e6), "MB")
