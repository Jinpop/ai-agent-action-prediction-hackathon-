#!/usr/bin/env python3
"""HOLDOUT_ONLY=1 제어흐름 테스트 (GPU/데이터/학습 없음, 순수 AST 정적검증 + 스켈레톤 시뮬레이션).
목적: 신규 HOLDOUT_ONLY 플래그가 요구동작 1~8을 만족하는지 실제 파일과 분기 로직으로 검증."""
import ast, sys, difflib

SRC = "open/scripts/colab_train_base2.py"      # HOLDOUT_ONLY 판
REF = "scratchpad/colab_dabf455d_ref.py"       # plan-review dabf455d 참조

src = open(SRC, encoding="utf-8").read()
tree = ast.parse(src)
lines = src.splitlines()
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))

# ---- A. AST 구조: 플래그·assert·config ----
check("A1_flag_parse", any(
    isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "HOLDOUT_ONLY" for t in n.targets)
    for n in ast.walk(tree)), "HOLDOUT_ONLY = os.environ.get(...)")
check("A2_assert_xor_refit", any(
    isinstance(n, ast.Assert) and "HOLDOUT_ONLY" in ast.dump(n) and "REFIT_ONLY" in ast.dump(n)
    for n in ast.walk(tree)))
# effective_config 딕셔너리 키
cfg_key = False
for n in ast.walk(tree):
    if isinstance(n, ast.Dict):
        for k in n.keys:
            if isinstance(k, ast.Constant) and k.value == "HOLDOUT_ONLY":
                cfg_key = True
check("A3_effective_config_key", cfg_key)

# ---- B. HOLDOUT_ONLY 종료 블록 위치/내용 ----
holdout_if = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                   and n.test.id == "HOLDOUT_ONLY"), None)
check("B1_holdout_if_exists", holdout_if is not None)
exit_line = gate_done = None
if holdout_if:
    for n in ast.walk(holdout_if):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "exit" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "sys"):
            exit_line = n.lineno
    gate_done = "GATE_DONE" in ast.dump(holdout_if)
check("B2_sys_exit_in_block", exit_line is not None)
check("B3_gate_done_written", bool(gate_done))

# ---- C. 라인 순서: holdout저장 < HOLDOUT_ONLY종료 < full-refit < zip ----
def call_line(pred):
    return next((n.lineno for n in ast.walk(tree) if pred(n)), None)
save_line = call_line(lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "save" and n.args and isinstance(n.args[0], ast.Constant)
                      and n.args[0].value == "holdout_idx3.npy")
# 배포 full-refit = `_, model_full, scaler_full = train_model(np.arange(...))` (PRETEXT의 train_model 431과 구분)
def _is_deploy_refit(n):
    if not isinstance(n, ast.Assign):
        return False
    names = [t.id for tgt in n.targets for t in ast.walk(tgt) if isinstance(t, ast.Name)]
    return "model_full" in names and isinstance(n.value, ast.Call) \
        and isinstance(n.value.func, ast.Name) and n.value.func.id == "train_model"
refit_line = next((n.lineno for n in ast.walk(tree) if _is_deploy_refit(n)), None)
zip_line = next((i for i, l in enumerate(lines, 1) if 'zp = "submit_base2.zip"' in l), None)
check("C0_landmarks_found", all([save_line, exit_line, refit_line, zip_line]),
      f"save={save_line} exit={exit_line} refit={refit_line} zip={zip_line}")
if all([save_line, exit_line, refit_line, zip_line]):
    check("C1_save_before_exit", save_line < exit_line, f"{save_line}<{exit_line}")
    check("C2_exit_before_full_refit", exit_line < refit_line, f"{exit_line}<{refit_line}")
    check("C3_full_refit_before_zip", refit_line < zip_line, f"{refit_line}<{zip_line}")

# ---- D. backward-compat: 추가된 라인이 전부 선언된 HOLDOUT_ONLY 관련뿐 ----
ref = open(REF, encoding="utf-8").read().splitlines()
added = [l[1:] for l in difflib.unified_diff(ref, lines, lineterm="")
         if l.startswith("+") and not l.startswith("+++")]
removed = [l[1:] for l in difflib.unified_diff(ref, lines, lineterm="")
           if l.startswith("-") and not l.startswith("---")]
markers = ["HOLDOUT_ONLY", "GATE_DONE", "_gf", "밴드 자격만", "sys.exit(0)", "홀드 Macro-F1", "생략, GATE_DONE"]
bad_add = [l for l in added if not any(m in l for m in markers)]
check("D1_only_declared_additions", not bad_add, f"예상밖 추가:{bad_add}")
check("D2_no_deletions", not removed, f"삭제된 라인:{removed}")

# ---- E. 스켈레톤 제어흐름 시뮬레이션 (실제 분기 구조 재현) ----
class _Exit(Exception):
    pass
def simulate(HOLDOUT_ONLY, REFIT_ONLY=False, SKIP_REFIT=False):
    c = []
    try:
        if REFIT_ONLY:
            c.append("refit_only_branch")
        else:
            c += ["train_model(tr,va)", "predict(va)", "save_holdout_probs3", "save_holdout_idx3"]
            if HOLDOUT_ONLY:
                c += ["write_GATE_DONE", "sys.exit(0)"]
                raise _Exit()
        if SKIP_REFIT:
            c.append("model_full=holdout_model")
        else:
            c.append("train_model(FULL)")
        c += ["build_SUB_DIR", "save_script.py+requirements.txt", "build_submit_base2.zip"]
    except _Exit:
        pass
    return c

ho = simulate(HOLDOUT_ONLY=True)                      # 요구: gate 발사 시나리오
check("E1_HO_saves_band_input", {"save_holdout_probs3", "save_holdout_idx3"} <= set(ho))
check("E2_HO_gate_done_and_exit", {"write_GATE_DONE", "sys.exit(0)"} <= set(ho))
check("E3_HO_no_full_train", "train_model(FULL)" not in ho)
check("E4_HO_no_model_sub", "build_SUB_DIR" not in ho)
check("E5_HO_no_script_reqs", "save_script.py+requirements.txt" not in ho)
check("E6_HO_no_zip", "build_submit_base2.zip" not in ho)

deflt = simulate(HOLDOUT_ONLY=False)                 # backward-compat 기본
check("E7_default_reaches_full_refit", "train_model(FULL)" in deflt)
check("E8_default_reaches_zip", "build_submit_base2.zip" in deflt)

skip = simulate(HOLDOUT_ONLY=False, SKIP_REFIT=True)  # SKIP_REFIT 의미 불변(사용자 교정: 여전히 zip)
check("E9_skiprefit_skips_full_train", "train_model(FULL)" not in skip)
check("E10_skiprefit_still_zips", "build_submit_base2.zip" in skip)

# ---- 보고 ----
print("=== HOLDOUT_ONLY 제어흐름 테스트 (AST 정적 + 스켈레톤 시뮬레이션) ===")
allok = True
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    allok = allok and ok
print(f"\n라인번호: holdout_save={save_line}  HOLDOUT_ONLY_exit={exit_line}  full_refit={refit_line}  zip={zip_line}")
print("HOLDOUT_ONLY=1 도달 콜:", simulate(True))
print("HOLDOUT_ONLY=0 도달 콜:", simulate(False))
print("\n전체:", "ALL PASS" if allok else "*** FAIL 있음 ***")
sys.exit(0 if allok else 1)
