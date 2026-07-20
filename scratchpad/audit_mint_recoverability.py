#!/usr/bin/env python3
"""Audit canonical mint coverage and point-in-time metadata recoverability.

This script is read-only. It treats a mint row as recoverable only when the
corresponding value is directly observed or uniquely implied by observed
session anchors and logged actions. Heuristic interpolation is reported
separately and never counted as exact recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "open" / "data"
EDIT_ACTIONS = {"edit_file", "write_file", "apply_patch"}
TOUCH_ACTIONS = {"read_file", "edit_file", "write_file"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def session_step(row_id: str) -> tuple[str, int]:
    session, raw_step = row_id.rsplit("-step_", 1)
    return session, int(raw_step.split("m", 1)[0])


def user_positions(history: list[dict[str, Any]]) -> list[int]:
    return [i for i, event in enumerate(history) if event.get("role") == "user"]


def canonical_target(mint_row: dict[str, Any]) -> tuple[str, int, int]:
    session, source_step = session_step(mint_row["id"])
    ui = int(mint_row["id"].rsplit("m", 1)[1])
    positions = user_positions(mint_row.get("_source_history", []))
    return session, source_step - len(positions) + ui, source_step


def action_after(history: list[dict[str, Any]], user_pos: int) -> dict[str, Any] | None:
    for event in history[user_pos + 1 :]:
        if event.get("role") == "user":
            return None
        if event.get("role") == "assistant_action":
            return event
    return None


def ci_result(event: dict[str, Any]) -> str | None:
    if event.get("name") != "run_tests":
        return None
    result = str(event.get("result_summary", ""))
    if re.search(r"PASS|green|passed", result, re.I):
        return "passed"
    if re.search(r"FAIL|red|error", result, re.I):
        return "failed"
    return None


def frozen(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def workspace(row: dict[str, Any]) -> dict[str, Any]:
    return ((row.get("session_meta") or {}).get("workspace") or {})


def replay_dirty(value: bool, event: dict[str, Any]) -> bool:
    return True if event.get("name") in EDIT_ACTIONS else value


def replay_open_files(value: list[str], event: dict[str, Any]) -> list[str]:
    result = list(value)
    if event.get("name") in TOUCH_ACTIONS:
        path = (event.get("args") or {}).get("path")
        if path and path not in result:
            result.append(path)
    return result


def load_npy_i8(path: Path) -> list[int]:
    """Read the project's simple NumPy v1.0, little-endian int64 holdout file."""
    with path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"not an npy file: {path}")
        major, minor = f.read(2)
        if (major, minor) != (1, 0):
            raise ValueError(f"unsupported npy version: {(major, minor)}")
        header_len = struct.unpack("<H", f.read(2))[0]
        header = f.read(header_len).decode("latin1")
        if "'<i8'" not in header and '"<i8"' not in header:
            raise ValueError(f"expected little-endian int64 npy: {header}")
        payload = f.read()
    return [value[0] for value in struct.iter_unpack("<q", payload)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    real = load_jsonl(DATA / "train.jsonl")
    mint1 = load_jsonl(DATA / "train_mint.jsonl")
    mint2 = load_jsonl(DATA / "train_mint2.jsonl")

    def input_key(row: dict[str, Any]) -> tuple[str, str]:
        return frozen(row.get("history") or []), row.get("current_prompt", "")

    real_input_keys = {input_key(row) for row in real}
    mint1_input_counts = Counter(input_key(row) for row in mint1)
    strict_text_overlap = sum(count for key, count in mint1_input_counts.items() if key in real_input_keys)

    real_by_session: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in real:
        session, step = session_step(row["id"])
        real_by_session[session][step] = row

    # Recover canonical step and action-event timelines from every overlapping
    # real history. Multiple observations of the same event are cross-checked.
    observed_events: dict[tuple[str, int], dict[str, Any]] = {}
    event_conflicts: Counter[str] = Counter()
    for row in real:
        session, source_step = session_step(row["id"])
        history = row.get("history") or []
        positions = user_positions(history)
        base_step = source_step - len(positions)
        for ui, pos in enumerate(positions):
            event = action_after(history, pos)
            if event is None:
                continue
            key = (session, base_step + ui)
            old = observed_events.get(key)
            if old is None:
                observed_events[key] = event
            elif frozen(old) != frozen(event):
                event_conflicts["full_event"] += 1
                if old.get("name") != event.get("name"):
                    event_conflicts["action_name"] += 1
                if ci_result(old) != ci_result(event):
                    event_conflicts["ci_result"] += 1

    # Annotate mint rows with their source history only in memory.
    canonical_rows: dict[tuple[str, int], dict[str, Any]] = {}
    alignment_bad = 0
    for row in mint1:
        session, source_step = session_step(row["id"])
        source = real_by_session[session][source_step]
        row["_source_history"] = source.get("history") or []
        target_session, target_step, _ = canonical_target(row)
        ui = int(row["id"].rsplit("m", 1)[1])
        pos = user_positions(row["_source_history"])[ui]
        if row.get("current_prompt") != row["_source_history"][pos].get("content"):
            alignment_bad += 1
        canonical_rows[(target_session, target_step)] = row

    mint2_keys: Counter[tuple[str, int]] = Counter()
    mint2_exact_inputs: Counter[tuple[str, str]] = Counter()
    mint2_longest: dict[tuple[str, int], int] = defaultdict(int)
    for row in mint2:
        session, source_step = session_step(row["id"])
        source = real_by_session[session][source_step]
        history = source.get("history") or []
        ui = int(row["id"].rsplit("m", 1)[1])
        key = (session, source_step - len(user_positions(history)) + ui)
        mint2_keys[key] += 1
        mint2_longest[key] = max(mint2_longest[key], len(row.get("history") or []))
        mint2_exact_inputs[(frozen(row.get("history") or []), row.get("current_prompt", ""))] += 1

    stable_paths = {
        "user_tier": lambda r: (r.get("session_meta") or {}).get("user_tier"),
        "language_pref": lambda r: (r.get("session_meta") or {}).get("language_pref"),
        "workspace.loc": lambda r: ((r.get("session_meta") or {}).get("workspace") or {}).get("loc"),
        "workspace.language_mix": lambda r: ((r.get("session_meta") or {}).get("workspace") or {}).get("language_mix"),
    }
    stable_summary: dict[str, dict[str, int]] = {}
    for name, getter in stable_paths.items():
        varying = 0
        for rows in real_by_session.values():
            if len({frozen(getter(row)) for row in rows.values()}) > 1:
                varying += 1
        stable_summary[name] = {
            "sessions": len(real_by_session),
            "varying_sessions": varying,
            "recoverable_mint_rows": len(mint1) if varying == 0 else 0,
        }

    turn_exact_real = 0
    turn_offsets: Counter[int] = Counter()
    for row in real:
        _, step = session_step(row["id"])
        turn = (row.get("session_meta") or {}).get("turn_index")
        if isinstance(turn, int):
            turn_offsets[turn - step] += 1
            turn_exact_real += int(turn == step)

    # Exact CI reconstruction. Forward replay starts at any earlier observed
    # row and requires every intervening action event. Backward anchoring is
    # valid only when no intervening run_tests can have changed CI.
    ci_exact = 0
    ci_methods: Counter[str] = Counter()
    ci_values: dict[tuple[str, int], str] = {}
    ci_unresolved_examples: list[str] = []
    ci_conflicting_derivations = 0
    for key in sorted(canonical_rows):
        session, target = key
        observed = real_by_session[session]
        derivations: list[tuple[str, str]] = []

        for anchor in sorted((step for step in observed if step < target), reverse=True):
            value = ((observed[anchor].get("session_meta") or {}).get("workspace") or {}).get("last_ci_status")
            complete = True
            for step in range(anchor, target):
                event = observed_events.get((session, step))
                if event is None:
                    complete = False
                    break
                update = ci_result(event)
                if update is not None:
                    value = update
            if complete and value is not None:
                derivations.append(("forward", value))
                break

        for anchor in sorted(step for step in observed if step > target):
            complete = True
            changed = False
            for step in range(target, anchor):
                event = observed_events.get((session, step))
                if event is None:
                    complete = False
                    break
                if event.get("name") == "run_tests":
                    changed = True
            if complete and not changed:
                value = ((observed[anchor].get("session_meta") or {}).get("workspace") or {}).get("last_ci_status")
                if value is not None:
                    derivations.append(("backward_no_test", value))
                break

        values = {value for _, value in derivations}
        if len(values) == 1:
            value = values.pop()
            ci_exact += 1
            ci_values[key] = value
            ci_methods["+".join(sorted({method for method, _ in derivations}))] += 1
        elif len(values) > 1:
            ci_conflicting_derivations += 1
        elif len(ci_unresolved_examples) < 12:
            ci_unresolved_examples.append(f"{session}-step_{target:02d}")

    # Current mint values are copied from a later source row. Compare them only
    # where the point-in-time CI has been proven exactly.
    current_ci_correct = 0
    fixed_ci_correct = 0
    ci_file = {row["id"]: row for row in load_jsonl(DATA / "train_mint_ci.jsonl")}
    for key, exact_value in ci_values.items():
        row = canonical_rows[key]
        current = ((row.get("session_meta") or {}).get("workspace") or {}).get("last_ci_status")
        fixed = ((ci_file[row["id"]].get("session_meta") or {}).get("workspace") or {}).get("last_ci_status")
        current_ci_correct += int(current == exact_value)
        fixed_ci_correct += int(fixed == exact_value)

    # Validate the state-transition rules on every pair of adjacent real rows.
    # In this dataset dirty is monotone (edit => true), while open_files appends
    # paths touched by read/edit/write and otherwise remains unchanged.
    transition_validation: Counter[str] = Counter()
    for session, rows in real_by_session.items():
        for step, before in rows.items():
            after = rows.get(step + 1)
            event = observed_events.get((session, step))
            if after is None or event is None:
                continue
            transition_validation["tested"] += 1
            dirty_expected = replay_dirty(bool(workspace(before).get("git_dirty")), event)
            dirty_actual = bool(workspace(after).get("git_dirty"))
            transition_validation["dirty_exact"] += int(dirty_expected == dirty_actual)
            open_expected = replay_open_files(list(workspace(before).get("open_files") or []), event)
            open_actual = list(workspace(after).get("open_files") or [])
            transition_validation["open_files_exact"] += int(open_expected == open_actual)

    def recover_dynamic(field: str) -> tuple[dict[tuple[str, int], Any], Counter[str], int]:
        recovered: dict[tuple[str, int], Any] = {}
        methods: Counter[str] = Counter()
        conflicts = 0
        for key in sorted(canonical_rows):
            session, target = key
            observed = real_by_session[session]
            derivations: list[tuple[str, Any]] = []

            for anchor in sorted((step for step in observed if step < target), reverse=True):
                if field == "git_dirty":
                    value: Any = bool(workspace(observed[anchor]).get(field))
                else:
                    value = list(workspace(observed[anchor]).get(field) or [])
                complete = True
                for step in range(anchor, target):
                    event = observed_events.get((session, step))
                    if event is None:
                        complete = False
                        break
                    value = replay_dirty(value, event) if field == "git_dirty" else replay_open_files(value, event)
                if complete:
                    derivations.append(("forward", value))
                    break

            # Conservative inverse reasoning. Dirty is exactly recoverable from
            # a later false anchor, or when no edit occurred. Open files are
            # copied backward only if no path-touch action occurred.
            for anchor in sorted(step for step in observed if step > target):
                events: list[dict[str, Any]] = []
                complete = True
                for step in range(target, anchor):
                    event = observed_events.get((session, step))
                    if event is None:
                        complete = False
                        break
                    events.append(event)
                if not complete:
                    continue
                if field == "git_dirty":
                    later = bool(workspace(observed[anchor]).get(field))
                    if not later:
                        derivations.append(("backward_false", False))
                    elif not any(event.get("name") in EDIT_ACTIONS for event in events):
                        derivations.append(("backward_no_edit", True))
                elif not any(event.get("name") in TOUCH_ACTIONS for event in events):
                    derivations.append(("backward_no_touch", list(workspace(observed[anchor]).get(field) or [])))
                break

            values = {frozen(value): value for _, value in derivations}
            if len(values) == 1:
                recovered[key] = next(iter(values.values()))
                methods["+".join(sorted({method for method, _ in derivations}))] += 1
            elif len(values) > 1:
                conflicts += 1
        return recovered, methods, conflicts

    dirty_values, dirty_methods, dirty_conflicts = recover_dynamic("git_dirty")
    open_values, open_methods, open_conflicts = recover_dynamic("open_files")
    full_rows = {row["id"]: row for row in load_jsonl(DATA / "train_mint_full.jsonl")}

    def recovery_accuracy(field: str, values: dict[tuple[str, int], Any]) -> dict[str, int]:
        source_correct = 0
        heuristic_correct = 0
        for key, exact in values.items():
            row = canonical_rows[key]
            source_value = workspace(row).get(field)
            heuristic_value = workspace(full_rows[row["id"]]).get(field)
            source_correct += int(source_value == exact)
            heuristic_correct += int(heuristic_value == exact)
        return {
            "source_copied_correct": source_correct,
            "train_mint_full_correct": heuristic_correct,
        }

    dirty_accuracy = recovery_accuracy("git_dirty", dirty_values)
    open_accuracy = recovery_accuracy("open_files", open_values)

    strict_keys = {
        key for key, row in canonical_rows.items() if input_key(row) not in real_input_keys
    }
    exact_meta_keys = set(ci_values) & set(dirty_values) & set(open_values)
    strict_exact_meta_keys = strict_keys & exact_meta_keys
    hidx_path = ROOT / "scratchpad" / "hidx.npy"
    holdout_sessions: set[str] = set()
    if hidx_path.exists():
        holdout_sessions = {
            session_step(real[index]["id"])[0] for index in load_npy_i8(hidx_path)
        }

    # Bracketing diagnostics for dynamic numeric/state fields. Equal values on
    # both sides are not called exact: hidden transitions can return to the same
    # value, and elapsed/budget values at the missing point remain unobserved.
    bracket: Counter[str] = Counter()
    numeric_interpolation_exact: Counter[str] = Counter()
    numeric_interpolation_total: Counter[str] = Counter()
    numeric_getters = {
        "budget_tokens_remaining": lambda r: (r.get("session_meta") or {}).get("budget_tokens_remaining"),
        "elapsed_session_sec": lambda r: (r.get("session_meta") or {}).get("elapsed_session_sec"),
    }
    temporal_monotonic: Counter[str] = Counter()
    for rows in real_by_session.values():
        ordered = [rows[step] for step in sorted(rows)]
        for left, right in zip(ordered, ordered[1:]):
            lm, rm = left.get("session_meta") or {}, right.get("session_meta") or {}
            temporal_monotonic["pairs"] += 1
            temporal_monotonic["budget_strict_decrease"] += int(
                lm.get("budget_tokens_remaining") > rm.get("budget_tokens_remaining")
            )
            temporal_monotonic["elapsed_strict_increase"] += int(
                lm.get("elapsed_session_sec") < rm.get("elapsed_session_sec")
            )
    for session, target in canonical_rows:
        steps = sorted(real_by_session[session])
        prev = max((step for step in steps if step < target), default=None)
        nxt = min((step for step in steps if step > target), default=None)
        bracket["has_prev"] += int(prev is not None)
        bracket["has_next"] += int(nxt is not None)
        bracket["has_both"] += int(prev is not None and nxt is not None)
        bracket["adjacent_prev"] += int(prev == target - 1)
        bracket["adjacent_next"] += int(nxt == target + 1)

        # Validate simple linear interpolation only on held-out real rows whose
        # immediate neighbours are both observed; this is diagnostic, not proof.
    for session, rows in real_by_session.items():
        for step, row in rows.items():
            if step - 1 not in rows or step + 1 not in rows:
                continue
            for name, getter in numeric_getters.items():
                left, actual, right = getter(rows[step - 1]), getter(row), getter(rows[step + 1])
                if all(isinstance(v, (int, float)) for v in (left, actual, right)):
                    numeric_interpolation_total[name] += 1
                    numeric_interpolation_exact[name] += int(actual * 2 == left + right)

    # Compare source-copied dynamic values with provably exact/known quantities.
    mint_turn_exact = 0
    for (session, target), row in canonical_rows.items():
        turn = (row.get("session_meta") or {}).get("turn_index")
        mint_turn_exact += int(turn == target)

    summary = {
        "coverage": {
            "mint1_rows": len(mint1),
            "mint1_canonical_unique": len(canonical_rows),
            "mint1_alignment_bad": alignment_bad,
            "mint2_rows": len(mint2),
            "mint2_canonical_unique": len(mint2_keys),
            "mint2_extra_window_variants": len(mint2) - len(mint2_keys),
            "mint2_max_variants_per_target": max(mint2_keys.values()),
            "mint2_exact_input_duplicates": sum(count - 1 for count in mint2_exact_inputs.values()),
            "mint1_is_longest_for_all_targets": all(
                len(row.get("history") or []) == mint2_longest[key]
                for key, row in canonical_rows.items()
            ),
            "mint1_exact_text_state_overlap_with_real": strict_text_overlap,
            "mint1_internal_exact_text_state_duplicates": sum(
                count - 1 for count in mint1_input_counts.values()
            ),
            "strict_text_unique_rows": len(mint1) - strict_text_overlap,
        },
        "event_timeline": {
            "observed_action_events": len(observed_events),
            "conflicts": dict(event_conflicts),
        },
        "stable_metadata": stable_summary,
        "turn_index": {
            "real_equals_id_step": turn_exact_real,
            "real_rows": len(real),
            "offset_distribution": dict(sorted(turn_offsets.items())),
            "mint_equals_canonical_step": mint_turn_exact,
            "mint_rows": len(mint1),
        },
        "ci": {
            "exactly_recoverable": ci_exact,
            "mint_rows": len(mint1),
            "methods": dict(ci_methods),
            "conflicting_derivations": ci_conflicting_derivations,
            "unresolved": len(mint1) - ci_exact,
            "unresolved_examples": ci_unresolved_examples,
            "source_copied_correct_on_exact_subset": current_ci_correct,
            "flawed_replay_correct_on_exact_subset": fixed_ci_correct,
        },
        "state_transition_validation": dict(transition_validation),
        "git_dirty": {
            "exactly_recoverable": len(dirty_values),
            "mint_rows": len(mint1),
            "methods": dict(dirty_methods),
            "conflicting_derivations": dirty_conflicts,
            **dirty_accuracy,
        },
        "open_files": {
            "exactly_recoverable": len(open_values),
            "mint_rows": len(mint1),
            "methods": dict(open_methods),
            "conflicting_derivations": open_conflicts,
            **open_accuracy,
        },
        "dynamic_brackets": dict(bracket),
        "temporal_monotonic_validation": dict(temporal_monotonic),
        "linear_interpolation_validation": {
            name: {
                "exact": numeric_interpolation_exact[name],
                "tested": numeric_interpolation_total[name],
            }
            for name in numeric_getters
        },
        "strict_interpretation": {
            "budget_elapsed_exactly_recoverable": 0,
            "budget_elapsed_reason": "unobserved point values; linear interpolation fails validation",
            "strict_text_rows": len(strict_keys),
            "strict_rows_with_exact_ci_dirty_open": len(strict_exact_meta_keys),
            "strict_rows_missing_any_ci_dirty_open": len(strict_keys - exact_meta_keys),
            "gate_train_strict_text_rows": sum(
                key[0] not in holdout_sessions for key in strict_keys
            ) if holdout_sessions else None,
            "gate_train_strict_exact_meta_rows": sum(
                key[0] not in holdout_sessions for key in strict_exact_meta_keys
            ) if holdout_sessions else None,
            "gate_excluded_holdout_strict_exact_meta_rows": sum(
                key[0] in holdout_sessions for key in strict_exact_meta_keys
            ) if holdout_sessions else None,
        },
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for section, values in summary.items():
            print(f"\n[{section}]")
            print(json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
