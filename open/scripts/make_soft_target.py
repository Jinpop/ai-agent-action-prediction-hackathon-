"""자기증류 teacher soft target 조립 (★07-13 temperature-matched 재작성 — v26 버그 픽스).

구 버전 버그(체크리스트20 item5): teacher를 softmax(logits) = **T=1**로 굽는데 colab student는
KD를 log_softmax(logits/T)·T=DISTILL_T(기본 2)로 적용 → teacher T=1 vs student T=2 불일치.
이 버전은 teacher를 **student와 동일 T**로 temper하고 T를 npz에 저장 → colab이 DISTILL_T==npz T를 assert.

사용법:
  make_soft_target.py --T 2.0 -o soft.npz oof_logits_seed74.npy oof_logits_seed79.npy oof_classic_probs.npy
  - 파일명에 'logits' 포함 → raw logits, temper = softmax(logits / T)
  - 파일명에 'probs'  포함 → 확률, temper = softmax(log(clip(p)) / T)  (동일 T로 재-소프트닝)
  - ids sidecar: 파일명 'logits'|'probs' → 'ids', 확장자 .npy → .json
  - 여러 teacher를 **공통 id 교집합(정렬)**에서 temper 후 평균. 모든 teacher는 feat.ACTIONS 열순서 가정.
출력 soft.npz: probs(N,14 float32) + ids(N) + temperature(scalar). colab_train_base2.py의 SOFT_TARGET= 로 사용.

주의: 이 스크립트는 조립만 한다. 커버리지·세션분리·in-sample금지·정렬·확률합1은 20_oof_distill 체크리스트 +
data-auditor(카드7 조건 'temperature/split/id/class-order 감사 PASS')로 학습 전 검증한다.
"""
import argparse
import json

import numpy as np


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _ids_path(npy_path):
    return npy_path.replace("logits", "ids").replace("probs", "ids").replace(".npy", ".json")


def _load_teacher(path, T):
    """return (ids:list, tempered_probs:np.ndarray[N,14]). logits/probs 자동판별 후 동일 T로 temper."""
    arr = np.load(path).astype(np.float64)
    ids = json.load(open(_ids_path(path)))
    assert len(ids) == arr.shape[0], f"{path}: ids {len(ids)} != rows {arr.shape[0]}"
    name = path.rsplit("/", 1)[-1]
    if "logits" in name:
        tempered = _softmax(arr / T)               # raw logits → matched-T softmax
        kind = "logits"
    elif "probs" in name:
        p = np.clip(arr, 1e-8, 1.0)
        tempered = _softmax(np.log(p) / T)         # 확률 → log 후 동일 T로 재-소프트닝
        kind = "probs"
    else:
        raise ValueError(f"{path}: 파일명에 'logits'/'probs' 없음 — teacher 종류 판별 불가")
    assert np.allclose(tempered.sum(1), 1.0, atol=1e-5) and np.isfinite(tempered).all(), \
        f"{path}: temper 후 확률합/finite 실패"
    print(f"  [{kind}] {name}: {arr.shape[0]}행, T={T} temper 완료 (평균 최대확률 {tempered.max(1).mean():.3f})")
    return ids, tempered


def main(paths, out, T):
    teachers = [_load_teacher(p, T) for p in paths]
    # 공통 id 교집합 (첫 teacher 순서 유지)
    common = set(teachers[0][0])
    for ids, _ in teachers[1:]:
        common &= set(ids)
    ordered = [i for i in teachers[0][0] if i in common]
    print(f"[soft] 공통 id {len(ordered)} (각 teacher: {[len(t[0]) for t in teachers]})")
    assert ordered, "공통 id 0 — teacher id 집합이 겹치지 않음"
    acc = np.zeros((len(ordered), teachers[0][1].shape[1]), dtype=np.float64)
    for ids, tp in teachers:
        idx = {i: r for r, i in enumerate(ids)}
        acc += tp[[idx[i] for i in ordered]]
    probs = (acc / len(teachers)).astype(np.float32)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5) and np.isfinite(probs).all()
    np.savez(out, probs=probs, ids=np.array(ordered), temperature=np.float32(T))
    print(f"[soft] {len(paths)}개 teacher 평균 -> {out}  probs{probs.shape} T={T}")
    print(f"[soft] 평균 최대확률 {probs.max(1).mean():.3f} (낮을수록 부드러움). ★colab DISTILL_T={T}로 학습해야 matched.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("teachers", nargs="+", help="oof_logits_*.npy / *_probs.npy (ids sidecar 필요)")
    ap.add_argument("--T", type=float, default=2.0, help="KD temperature (colab DISTILL_T와 반드시 일치)")
    ap.add_argument("-o", "--out", default="soft.npz")
    a = ap.parse_args()
    main([p for p in a.teachers if p.endswith(".npy")], a.out, a.T)
