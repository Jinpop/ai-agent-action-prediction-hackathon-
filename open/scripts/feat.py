"""공통 feature 엔지니어링 모듈 (학습 train.py 와 추론 script.py 가 함께 사용).

baseline 은 current_prompt 문자열 하나만 썼지만, 여기서는
 - current_prompt + 최근 history 발화 텍스트  -> TF-IDF (+ SVD 로 축소)
 - session_meta (요금제/워크스페이스/토큰예산/turn 등)
 - history 로부터 파생한 행동 통계 (직전 행동, 행동별 빈도 등)
를 모두 사용한다. 추론 때 학습과 '동일한' feature 를 만들어야 하므로
컬럼 순서를 고정하기 위한 상수(ACTIONS, LANGS 등)를 여기에 둔다.
"""
import json
import os

import numpy as np
import pandas as pd

# ★META-N (campaign official-meta-wave1): 누락 language_mix 5키 + langpref_mixed 를 숫자 메타에 추가.
# META_NUM_EXT=1 일 때만 활성(default 0 = 기존 119d 와 byte-identical). 배포는 =1로 켜되 기존 멤버는
# prep 컬럼 reindex 로 119 유지(신규 6키 drop) → 하위호환. META-N 멤버만 125d.
META_NUM_EXT = os.environ.get("META_NUM_EXT", "0") == "1"
META_N_EXTRA_LANGS = ["kt", "vue", "swift", "ipynb", "tf"]

# 예측 대상 14개 행동 클래스 (history 의 직전 행동 인코딩에도 재사용)
ACTIONS = [
    "edit_file", "grep_search", "read_file", "glob_pattern", "respond_only",
    "run_bash", "apply_patch", "run_tests", "list_directory", "ask_user",
    "plan_task", "lint_or_typecheck", "write_file", "web_search",
]

# language_mix / language_pref 에서 고정으로 뽑을 언어 집합
LANGS = [
    "py", "ts", "tsx", "js", "jsx", "css", "json", "md", "yaml", "yml",
    "go", "rs", "java", "html", "dockerfile", "sql", "sh", "toml", "cpp", "c",
]

CI_STATUSES = ["passed", "failed", "none", "running", "unknown"]
TIERS = ["free", "pro", "enterprise", "team"]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_str(x):
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def extract_current_prompt(sample):
    """baseline 과 동일하게 current_prompt 문자열만 추출."""
    return _safe_str(sample.get("current_prompt", ""))


def build_text(sample):
    """TF-IDF 입력 텍스트: 현재 발화 + 최근 user 발화 + 행동 시퀀스 + 열린 파일 컨텍스트."""
    parts = []
    hist = sample.get("history") or []
    user_turns = [h.get("content", "") for h in hist
                  if isinstance(h, dict) and h.get("role") == "user"]
    for t in user_turns[-3:]:
        parts.append(_safe_str(t))
    # 과거 행동 전체 시퀀스(최근 8개)를 순서 보존 토큰으로 (전이 패턴 포착)
    actions = [h.get("name") for h in hist
               if isinstance(h, dict) and h.get("role") == "assistant_action"]
    for a in actions[-8:]:
        parts.append("seq_" + _safe_str(a))
    # 열린 파일의 basename/확장자 토큰 (워크스페이스 컨텍스트)
    ws = (sample.get("session_meta") or {}).get("workspace") or {}
    for pth in (ws.get("open_files") or [])[:6]:
        pth = _safe_str(pth)
        base = pth.replace("/", "_").replace(".", "_")
        parts.append("openf_" + base)
        if "." in pth:
            parts.append("opext_" + pth.rsplit(".", 1)[-1])
    parts.append(_safe_str(sample.get("current_prompt", "")))
    return " \n ".join(parts).strip()


def build_meta_row(sample):
    """session_meta + history 파생 통계를 dict(고정 키) 로 반환."""
    d = {}
    meta = sample.get("session_meta") or {}
    ws = meta.get("workspace") or {}

    # --- 요금제 / 언어 선호 (one-hot) ---
    tier = _safe_str(meta.get("user_tier")).lower()
    for t in TIERS:
        d[f"tier_{t}"] = 1.0 if tier == t else 0.0
    lang_pref = _safe_str(meta.get("language_pref")).lower()
    d["langpref_en"] = 1.0 if lang_pref == "en" else 0.0

    # --- 워크스페이스 ---
    d["loc"] = float(ws.get("loc") or 0)
    d["git_dirty"] = 1.0 if ws.get("git_dirty") else 0.0
    open_files = ws.get("open_files") or []
    d["n_open_files"] = float(len(open_files))
    d["has_open_files"] = 1.0 if open_files else 0.0
    ci = _safe_str(ws.get("last_ci_status")).lower() or "none"
    for s in CI_STATUSES:
        d[f"ci_{s}"] = 1.0 if ci == s else 0.0
    lang_mix = ws.get("language_mix") or {}
    for lg in LANGS:
        d[f"lang_{lg}"] = float(lang_mix.get(lg) or 0.0)
    if META_NUM_EXT:  # META-N: 누락 language_mix 5키 + langpref_mixed (119 -> 125)
        for lg in META_N_EXTRA_LANGS:
            d[f"lang_{lg}"] = float(lang_mix.get(lg) or 0.0)
        d["langpref_mixed"] = 1.0 if lang_pref == "mixed" else 0.0

    # --- 예산 / 진행도 ---
    d["budget_tokens"] = float(meta.get("budget_tokens_remaining") or 0)
    d["turn_index"] = float(meta.get("turn_index") or 0)
    d["elapsed_sec"] = float(meta.get("elapsed_session_sec") or 0)

    # --- current_prompt 길이 ---
    cp = _safe_str(sample.get("current_prompt", ""))
    cp_l = cp.lower()
    d["cp_chars"] = float(len(cp))
    d["cp_words"] = float(len(cp.split()))
    d["cp_has_question"] = 1.0 if "?" in cp else 0.0
    # 현재 발화 키워드 플래그 (행동과 직결되는 강한 신호)
    KW = {
        "kw_test": ("test", "spec", "pytest", "unit test", "coverage"),
        "kw_run": ("run", "execute", "build", "compile", "start the", "launch"),
        "kw_fix": ("fix", "bug", "error", "fail", "broken", "crash", "wrong"),
        # --- 탐색 4형제 정밀 분리 ---
        "kw_read": ("open", "read", "show me", "look at", "pull up", "view",
                    "content of", "see the", "cat "),
        "kw_grep": ("grep", "search for", "occurrenc", "usages", "references to",
                    "where is", "where do", "look for", "find all", "find the string",
                    "find where", "instances of", "calls to"),
        "kw_glob": ("find files", "glob", "files named", "which files", "matching",
                    "files matching", "*.", "file pattern", "all the ", "under the"),
        "kw_list": ("list", "ls ", "what's in", "contents of", "files in",
                    "directory", "folder", "tree of"),
        "kw_write": ("write", "create", "add a new", "new file", "scaffold", "generate a"),
        "kw_edit": ("edit", "change", "update", "patch", "modify", "refactor",
                    "rename", "tweak", "adjust", "replace"),
        "kw_plan": ("plan", "steps", "how should", "approach", "strategy",
                    "outline", "break down", "figure out how"),
        "kw_install": ("install", "pip", "npm", "dependency", "package", "requirements"),
        "kw_git": ("commit", "git", "push", "branch", "merge", "stage", "diff"),
        "kw_ask": ("should i", "which ", "do you want", "?", "confirm", "not sure",
                   "clarify", "what do you"),
        "kw_lint": ("lint", "type check", "typecheck", "mypy", "flake", "format", "types"),
    }
    for key, words in KW.items():
        d[key] = 1.0 if any(w in cp_l for w in words) else 0.0

    # --- 정규식/문자 신호 (탐색 4형제 구분에 강함) ---
    import re
    d["has_glob_star"] = 1.0 if ("*." in cp or "*/" in cp or "**" in cp) else 0.0
    d["has_file_ext"] = 1.0 if re.search(
        r"\.(py|ts|tsx|js|jsx|css|json|md|ya?ml|go|rs|java|html|sql|sh|toml|cpp|c|h)\b",
        cp_l) else 0.0
    d["has_path_slash"] = 1.0 if "/" in cp else 0.0
    d["has_quote"] = 1.0 if ('"' in cp or "'" in cp or "`" in cp) else 0.0
    d["mentions_file_word"] = 1.0 if "file" in cp_l else 0.0
    d["mentions_dir_word"] = 1.0 if ("dir" in cp_l or "folder" in cp_l) else 0.0
    d["n_dots"] = float(cp.count("."))

    # --- 상태 인지 신호 (탐색 4형제의 진짜 구분자) ---
    # 프롬프트가 '이미 열린 파일'을 가리키나 -> read_file 신호
    open_bases = set()
    for p in open_files:
        b = _safe_str(p).rsplit("/", 1)[-1].lower()
        if b:
            open_bases.add(b)
            open_bases.add(b.rsplit(".", 1)[0])
    d["prompt_hits_open_file"] = 1.0 if any(
        b and len(b) >= 3 and b in cp_l for b in open_bases) else 0.0
    # 구체적 파일명(word.ext) 언급 수 -> glob/read 신호
    fnames = re.findall(
        r"[\w\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|json|ya?ml|toml|md|css|html|"
        r"sql|sh|gradle|txt|cfg|ini|c|h|cpp|tf|env|lock|sum|mod)\b", cp_l)
    d["n_filenames_in_prompt"] = float(len(fnames))
    d["has_filename"] = 1.0 if fnames else 0.0
    # grep: 심볼/텍스트가 '어디' 쓰이나 (코드 전반 검색)
    d["sig_where"] = 1.0 if any(w in cp_l for w in (
        "어디", "where", "used", "쓰이", "쓰는", "정의", "defined", "호출",
        "calls", "참조", "references", "usages", "occurrenc", "흩어")) else 0.0
    # list: 폴더/구조에 '뭐가 있나'
    d["sig_whatsin"] = 1.0 if any(w in cp_l for w in (
        "뭐뭐 있", "뭐 있", "있는지", "있나", "what's in", "구조", "structure",
        "셋업", "order of", "export", "들어있", "뭐가 있")) else 0.0
    # glob: 이름 패턴으로 파일 찾기
    d["sig_pattern"] = 1.0 if any(w in cp_l for w in (
        "패턴", "every ", "모든", "전부", "싹", "matching", "glob", "*.",
        "찾아", "find every", "all the")) else 0.0

    # --- history 파생 ---
    hist = sample.get("history") or []
    actions = [h.get("name") for h in hist
               if isinstance(h, dict) and h.get("role") == "assistant_action"]
    d["n_history"] = float(len(hist))
    d["n_actions"] = float(len(actions))
    d["n_user_turns"] = float(sum(
        1 for h in hist if isinstance(h, dict) and h.get("role") == "user"))
    # 행동별 빈도
    for a in ACTIONS:
        d[f"cnt_{a}"] = float(actions.count(a))
    # 직전 행동 one-hot
    last = actions[-1] if actions else None
    for a in ACTIONS:
        d[f"last_{a}"] = 1.0 if last == a else 0.0
    d["last_is_none"] = 1.0 if last is None else 0.0
    # 2번째 직전 행동 one-hot (행동 전이 패턴 포착)
    prev2 = actions[-2] if len(actions) >= 2 else None
    for a in ACTIONS:
        d[f"prev2_{a}"] = 1.0 if prev2 == a else 0.0
    # 직전 행동 == 2번째 직전 (같은 행동 반복 신호)
    d["last_eq_prev2"] = 1.0 if (last is not None and last == prev2) else 0.0
    # 행동 밀도 (턴당 행동 수)
    ti = float(meta.get("turn_index") or 0)
    d["actions_per_turn"] = float(len(actions)) / (ti + 1.0)

    # 직전 action 의 args 신호 (n_files, target=all)
    last_args = {}
    for h in reversed(hist):
        if isinstance(h, dict) and h.get("role") == "assistant_action":
            last_args = h.get("args") or {}
            break
    d["last_n_files"] = float(last_args.get("n_files") or 0)
    tgt = _safe_str(last_args.get("target")).lower()
    d["last_target_all"] = 1.0 if tgt == "all" else 0.0
    # 직전 결과 요약의 신호 (PASS/FAIL/error)
    last_result = ""
    for h in reversed(hist):
        if isinstance(h, dict) and h.get("role") == "assistant_action":
            last_result = _safe_str(h.get("result_summary")).lower()
            break
    d["last_pass"] = 1.0 if "pass" in last_result else 0.0
    d["last_fail"] = 1.0 if ("fail" in last_result or "error" in last_result) else 0.0
    return d


def build_meta_frame(samples, columns=None):
    """샘플 리스트 -> 고정 컬럼 순서의 dense DataFrame.

    columns 를 주면 그 순서/집합으로 reindex (추론 시 학습 컬럼에 정렬).
    """
    rows = [build_meta_row(s) for s in samples]
    df = pd.DataFrame(rows).astype(np.float32)
    if columns is not None:
        df = df.reindex(columns=columns, fill_value=0.0).astype(np.float32)
    return df


def session_of(_id):
    """'sess_..._-step_08' -> 세션 id (GroupKFold 용)."""
    return _id.rsplit("-step", 1)[0]
