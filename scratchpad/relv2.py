"""Relational v2 — replay-derived state features (12), classic-only info axis.
All computable from (current_prompt, history, workspace) alone — deploy-identical.
No overlap with 119 meta (cnt_/last_/prev2_/last_pass/fail/prompt_hits_open_file)
nor rel_* 9 (glob/nfile/wasread/pastpath/lastsib/lastfail/nopen/dirword/searchword).
"""
import re as _re

_FILE_RE = _re.compile(r'[\w./-]+\.\w{1,5}')
_NUM_RE = _re.compile(r'(\d+)')
EDIT_ACTS = {"apply_patch", "edit_file", "write_file"}
SEARCH_ACTS = {"read_file", "grep_search", "glob_pattern", "list_directory"}

REL2_COLS = ["r2_edits_since_test", "r2_searches_since_edit", "r2_distinct_files",
             "r2_prompt_file_edited", "r2_last_grep_matches", "r2_zero_match_last",
             "r2_same_streak", "r2_steps_since_fail", "r2_never_tested",
             "r2_frac_search", "r2_prompt_overlap_prev", "r2_last_num"]

def _sr(x): return x if isinstance(x, str) else ("" if x is None else str(x))

def relfeats2(s):
    cp = _sr(s.get("current_prompt", "")); cpl = cp.lower()
    h = s.get("history") or []
    acts = []          # (name, path, result)
    prev_user = None
    for e in h:
        if not isinstance(e, dict):
            continue
        if e.get("role") == "user":
            prev_user = _sr(e.get("content", ""))
        elif e.get("role") == "assistant_action":
            a = e.get("args") or {}
            acts.append((e.get("name"), _sr(a.get("path") or a.get("target") or ""),
                         _sr(e.get("result_summary", ""))))
    n = len(acts)
    # 1) edits since last run_tests
    edits_since_test = 0
    for nm, _, _ in reversed(acts):
        if nm == "run_tests":
            break
        if nm in EDIT_ACTS:
            edits_since_test += 1
    # 2) searches since last edit
    searches_since_edit = 0
    for nm, _, _ in reversed(acts):
        if nm in EDIT_ACTS:
            break
        if nm in SEARCH_ACTS:
            searches_since_edit += 1
    # 3) distinct files touched
    files = set(p.lower() for _, p, _ in acts if p and "." in p)
    # 4) prompt mentions a file that was EDITED (wasread는 rel_에 이미 있음)
    edited = set()
    for nm, p, _ in acts:
        if nm in EDIT_ACTS and p:
            edited.add(p.lower())
    pf_edit = 1.0 if any(p.split("/")[-1].split(".")[0] in cpl for p in edited if p) else 0.0
    # 5,6) last grep match count / zero-match flag
    grep_m = -1.0
    for nm, _, r in reversed(acts):
        if nm == "grep_search":
            m = _NUM_RE.search(r)
            grep_m = float(m.group(1)) if m else 0.0
            break
    zero_match = 1.0 if grep_m == 0.0 else 0.0
    # 7) trailing same-action streak
    streak = 0
    if acts:
        last_nm = acts[-1][0]
        for nm, _, _ in reversed(acts):
            if nm == last_nm:
                streak += 1
            else:
                break
    # 8) steps since last FAIL/error result (n+1 if never)
    since_fail = float(n + 1)
    for i, (_, _, r) in enumerate(reversed(acts)):
        rl = r.lower()
        if "fail" in rl or "error" in rl:
            since_fail = float(i)
            break
    # 9) never ran tests
    never_tested = 1.0 if not any(nm == "run_tests" for nm, _, _ in acts) else 0.0
    # 10) search-family fraction
    frac_search = (sum(1 for nm, _, _ in acts if nm in SEARCH_ACTS) / n) if n else 0.0
    # 11) prompt token overlap with previous user prompt (연속작업 vs 새주제)
    ov = 0.0
    if prev_user:
        a = set(_re.findall(r"[\w가-힣]+", cpl))
        b = set(_re.findall(r"[\w가-힣]+", prev_user.lower()))
        if a and b:
            ov = len(a & b) / max(len(a), 1)
    # 12) leading number in last result (grep 외 일반화: tests passed 수 등)
    last_num = 0.0
    if acts:
        m = _NUM_RE.search(acts[-1][2])
        last_num = float(m.group(1)) if m else 0.0
    return [float(edits_since_test), float(searches_since_edit), float(len(files)),
            pf_edit, grep_m, zero_match, float(streak), since_fail, never_tested,
            float(frac_search), ov, min(last_num, 999.0)]
