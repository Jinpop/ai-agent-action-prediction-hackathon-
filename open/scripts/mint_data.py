"""history 민팅 v2: train 각 세션의 가장 깊은 row의 history에서
새 (상태→행동) 학습예제 합성. 라벨 추출 정확도 99.99% 검증됨.

v2 변경:
- 중복 제거: (세션, 프롬프트 내용) 기준 — 실제 row와 같은 상태 제외
  (history가 롤링 윈도우라 길이 시그니처는 불일치함)
- turn_index 보정: 소스 turn_index - (절단된 이후의 user턴 수) 로 추정, 최소 0
출력: data/train_mint.jsonl + train_mint_labels.csv
"""
import copy
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feat
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
samples = feat.load_jsonl(os.path.join(DATA, "train.jsonl"))
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
lab = dict(zip(labels["id"], labels["action"]))

def sess_step(i):
    s, st = i.rsplit("-step_", 1)
    return s, int(st)

by_sess = defaultdict(dict)
for smp in samples:
    s, st = sess_step(smp["id"])
    by_sess[s][st] = smp

mint, mint_lab = [], {}
skipped_dup = 0
for s, steps in by_sess.items():
    real_prompts = {r.get("current_prompt", "") for r in steps.values()}
    seen_mint = set()
    for st_key in sorted(steps):
      src = steps[st_key]
      h = src.get("history") or []
      src_ti = (src.get("session_meta") or {}).get("turn_index", 0)
      upos = [p for p, e in enumerate(h) if isinstance(e, dict) and e.get("role") == "user"]
      n_u = len(upos)
      for ui, p in enumerate(upos):
        prompt = h[p].get("content", "")
        if prompt in real_prompts or prompt in seen_mint:
            skipped_dup += 1
            continue
        label = None
        for q in range(p + 1, len(h)):
            if isinstance(h[q], dict) and h[q].get("role") == "assistant_action":
                label = h[q].get("name")
                break
            if isinstance(h[q], dict) and h[q].get("role") == "user":
                break
        if not label:
            continue
        seen_mint.add(prompt)
        r = copy.deepcopy(src)
        r["id"] = f"{s}-step_{st_key:02d}m{ui:02d}"
        r["history"] = h[:p]
        r["current_prompt"] = prompt
        if isinstance(r.get("session_meta"), dict):
            removed_after = n_u - ui  # 이 턴 포함 이후의 user턴 수
            r["session_meta"]["turn_index"] = max(0, src_ti - removed_after)
        mint.append(r)
        mint_lab[r["id"]] = label

print(f"민팅 신규: {len(mint)}개 (+{len(mint)/len(samples)*100:.0f}%), 실제상태 중복 제외 {skipped_dup}")
hist_len = Counter(len(r["history"]) for r in mint)
print("history 길이 분포(상위):", dict(sorted(hist_len.items())[:6]))
print("라벨 분포(상위 8):", Counter(mint_lab.values()).most_common(8))

with open(os.path.join(DATA, "train_mint.jsonl"), "w") as f:
    for r in mint:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(DATA, "train_mint_labels.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "action"])
    for k, v in mint_lab.items():
        w.writerow([k, v])
print("저장 완료")
