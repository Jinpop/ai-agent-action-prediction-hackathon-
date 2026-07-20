"""카드3 완전matched 대조군: train_mint_exact{,_gate}.jsonl에 target_key 추가.
exact는 target당 1행(최장 window)이라 각 행이 고유 target → target_key=f'{session}#{turn_index}'.
이렇게 하면 mint2b와 동일한 A3/B6 TARGET_BALANCE=1 체인으로 돌 수 있고(가중 1/1=1),
유일 변경변수 = 데이터(단일윈도우 exact vs balanced rolling-window). labels는 원본 재사용."""
import json, sys, shutil
sys.path.insert(0, "open/scripts")
import feat
for stem in ["train_mint_exact", "train_mint_exact_gate"]:
    rows = [json.loads(l) for l in open(f"open/data/{stem}.jsonl")]
    out_stem = stem.replace("train_mint_exact", "train_mint1exact_tb")
    keys = []
    for r in rows:
        sess = feat.session_of(r["id"])
        tk = f"{sess}#{(r.get('session_meta') or {}).get('turn_index')}"
        r["target_key"] = tk; keys.append(tk)
    assert len(set(keys)) == len(keys), f"{stem}: target_key 중복! (exact는 target당 1행이어야)"
    with open(f"open/data/{out_stem}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(f"open/data/{stem}_labels.csv", f"open/data/{out_stem}_labels.csv")
    print(f"{out_stem}: {len(rows)}행, target_key 고유 {len(set(keys))} (중복0), labels 복사")
