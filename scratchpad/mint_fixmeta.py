#!/usr/bin/env python3
"""Reconstruct point-in-time workspace metadata for mint examples.

Two variants:
  A (ci):   fix ONLY last_ci_status via run_tests replay.
  B (full): fix ci + git_dirty (approx) + open_files (heuristic, count-preserving).

Rationale (measured):
  - last_ci_status: reliably reconstructable (run_tests -> passed/failed, none if untested).
  - git_dirty: True after any edit/patch/write; commit resets invisible (only 2 git-status
    in 20k run_bash) -> approximation only.
  - open_files: read/edit/write carry `path`; apply_patch hides filenames (n_files only).
    IDE tab set (usually 0-2) is decoupled from action log -> heuristic: distinct
    recently-touched paths, most-recent-first, capped at the session's ORIGINAL open_files
    count (preserves the count distribution, fixes identity only).
"""
import json, re, sys, copy

IN = "open/data/train_mint.jsonl"
OUT_CI = "open/data/train_mint_ci.jsonl"
OUT_FULL = "open/data/train_mint_full.jsonl"

EDIT_ACTS = {"apply_patch", "edit_file", "write_file"}
TOUCH_ACTS = {"read_file", "edit_file", "write_file"}  # apply_patch has no path


def replay(history):
    """Return (ci_status, git_dirty, touched_paths_recent_first) from truncated history."""
    ci = "none"
    dirty = False
    touched = []  # most-recent LAST; we reverse at end
    for e in history:
        n = e.get("name")
        if not n:
            continue
        res = str(e.get("result_summary", ""))
        args = e.get("args", {}) or {}
        if n == "run_tests":
            if re.search(r"PASS|green|passed", res, re.I):
                ci = "passed"
            elif re.search(r"FAIL|red|error", res, re.I):
                ci = "failed"
        if n in EDIT_ACTS:
            dirty = True
        if n in TOUCH_ACTS:
            p = args.get("path")
            if p:
                if p in touched:
                    touched.remove(p)
                touched.append(p)
    touched_recent_first = list(reversed(touched))
    return ci, dirty, touched_recent_first


def main():
    rows = [json.loads(l) for l in open(IN)]
    fci = open(OUT_CI, "w")
    ffull = open(OUT_FULL, "w")
    # mismatch stats vs original final-state metadata
    n = 0
    ci_mm = 0
    dirty_mm = 0
    of_mm = 0
    for r in rows:
        n += 1
        hist = r.get("history", [])
        ci, dirty, touched = replay(hist)
        orig_w = r.get("session_meta", {}).get("workspace", {})
        orig_ci = orig_w.get("last_ci_status")
        orig_dirty = orig_w.get("git_dirty")
        orig_of = list(orig_w.get("open_files", []))
        cap = len(orig_of)  # preserve count
        new_of = touched[:cap] if cap else []

        if ci != orig_ci:
            ci_mm += 1
        if dirty != orig_dirty:
            dirty_mm += 1
        if set(new_of) != set(orig_of):
            of_mm += 1

        # variant A: ci only
        ra = copy.deepcopy(r)
        ra["session_meta"]["workspace"]["last_ci_status"] = ci
        fci.write(json.dumps(ra, ensure_ascii=False) + "\n")

        # variant B: ci + dirty + open_files
        rb = copy.deepcopy(r)
        w = rb["session_meta"]["workspace"]
        w["last_ci_status"] = ci
        w["git_dirty"] = dirty
        w["open_files"] = new_of
        ffull.write(json.dumps(rb, ensure_ascii=False) + "\n")

    fci.close()
    ffull.close()
    print(f"mint examples: {n}")
    print(f"ci_status mismatch vs final-state: {ci_mm}/{n} = {ci_mm/n:.1%}")
    print(f"git_dirty mismatch: {dirty_mm}/{n} = {dirty_mm/n:.1%}")
    print(f"open_files (set) mismatch: {of_mm}/{n} = {of_mm/n:.1%}")
    print(f"wrote {OUT_CI}, {OUT_FULL}")


if __name__ == "__main__":
    main()
