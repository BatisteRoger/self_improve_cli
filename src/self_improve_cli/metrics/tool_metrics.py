"""Tool usage metrics — deterministic call patterns over a trace.

Pure functions over canonical Run objects. No SDK dependency, no LLM calls:
deterministic and recomputable from sanitized data.

The metrics are raw facts (counts, ratios, deltas). Interpreting them
("excessive granularity", "context bloat") is the job of the Analyst Agent
that consumes this file, not of this module.

APPROXIMATE: heuristics are provider- and agent-dependent signals, not exact
measurements. See notes in each section.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from self_improve_cli.domain import Run, RunType
from self_improve_cli.representations import significant_runs

# --- Configurable heuristics ------------------------------------------------

_TARGET_KEY_CANDIDATES = ("path", "file", "filename", "file_path", "slug", "target", "uri", "url")
_TARGET_KEY_IGNORE = ("type", "run_id", "id", "tool_call_id")
_TARGET_FALLBACK_MAX_LEN = 100
_MODIFY_NAME_PATTERN = ("edit", "write", "patch", "update", "replace", "set")
_MODIFY_ARG_HINTS = ("old_string", "new_string", "content", "patch")
_REPEAT_THRESHOLD = 3


# --- Helpers ----------------------------------------------------------------


def _tool_runs(sig: list[Run]) -> list[Run]:
    """Tool-type runs from an already-filtered significant-run list."""
    return [r for r in sig if r.run_type == RunType.TOOL]


def _tool_args(run: Run) -> dict[str, Any]:
    """Extract the args dict from a tool run's inputs."""
    inputs = run.inputs or {}
    if isinstance(inputs, dict):
        for key in ("input", "args", "inputs"):
            val = inputs.get(key)
            if isinstance(val, dict):
                return val
        return inputs
    return {}


# --- 1. Tool call frequency & sequencing ------------------------------------


def tool_call_counts(runs: list[Run]) -> dict[str, int]:
    """Count tool calls by tool name."""
    sig = significant_runs(runs)
    return dict(Counter(r.name for r in _tool_runs(sig)))


def extract_target_key(tool_name: str, args: dict[str, Any]) -> str | None:
    """Heuristically extract a target identifier from tool args.

    Prefers common path-like keys. Falls back to a short, identifier-like
    string arg (no newline, <= 100 chars, no internal spaces) so we don't
    mistake prose (queries, messages) for a target. Returns None if nothing
    suitable is found.
    """
    for key in _TARGET_KEY_CANDIDATES:
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    for k, v in args.items():
        if k in _TARGET_KEY_IGNORE:
            continue
        if (
            isinstance(v, str)
            and v
            and len(v) <= _TARGET_FALLBACK_MAX_LEN
            and "\n" not in v
            and " " not in v.strip()
        ):
            return v
    return None


def repeated_same_target(
    runs: list[Run], threshold: int = _REPEAT_THRESHOLD
) -> dict[str, list[tuple[str, int]]]:
    """Find tools called repeatedly on the same target.

    Returns {tool_name: [(target, count), ...]} sorted by count descending,
    keeping only entries with count >= threshold.
    """
    sig = significant_runs(runs)
    counts: dict[str, dict[str, int]] = {}
    for run in _tool_runs(sig):
        target = extract_target_key(run.name, _tool_args(run))
        if target is None:
            continue
        counts.setdefault(run.name, {})
        counts[run.name][target] = counts[run.name].get(target, 0) + 1

    result: dict[str, list[tuple[str, int]]] = {}
    for name, targets in counts.items():
        filtered = sorted(
            ((t, c) for t, c in targets.items() if c >= threshold),
            key=lambda x: -x[1],
        )
        if filtered:
            result[name] = filtered
    return result


# --- 2. Modify-call granularity ---------------------------------------------


def _is_modify_tool(tool_name: str, args: dict[str, Any]) -> bool:
    """Heuristic: does this tool modify content?"""
    name_lower = tool_name.lower()
    if any(pat in name_lower for pat in _MODIFY_NAME_PATTERN):
        return True
    return any(hint in args for hint in _MODIFY_ARG_HINTS)


def _count_changed_units(args: dict[str, Any]) -> int:
    """Estimate units changed by a modify call.

    Default unit is the line (max of old/new line counts), suitable for
    text/code edits. For tools that don't carry old/new strings, falls back
    to a size proxy. The unit definition is a heuristic; the Analyst Agent
    interprets.
    """
    old = args.get("old_string") or args.get("old") or ""
    new = args.get("new_string") or args.get("new") or args.get("content") or ""
    if not old and not new:
        return max(1, len(json.dumps(args, default=str)) // 80)
    old_lines = len(old.splitlines()) if old else 0
    new_lines = len(new.splitlines()) if new else 0
    return max(old_lines, new_lines, 1)


def modify_granularity(runs: list[Run]) -> dict[str, Any]:
    """Granularity metrics for modify-type tool calls.

    Returns total_modify_calls, total_units_changed, modify_efficiency
    (units per call), and a per_tool breakdown.
    """
    sig = significant_runs(runs)
    per_tool: dict[str, dict[str, int]] = {}
    total_calls = 0
    total_units = 0

    for run in _tool_runs(sig):
        args = _tool_args(run)
        if not _is_modify_tool(run.name, args):
            continue
        units = _count_changed_units(args)
        total_calls += 1
        total_units += units
        entry = per_tool.setdefault(run.name, {"calls": 0, "units": 0})
        entry["calls"] += 1
        entry["units"] += units

    result: dict[str, Any] = {
        "total_modify_calls": total_calls,
        "total_units_changed": total_units,
        "modify_efficiency": round(total_units / total_calls, 2) if total_calls else 0.0,
        "per_tool": {},
    }
    for name, entry in per_tool.items():
        result["per_tool"][name] = {
            "calls": entry["calls"],
            "units": entry["units"],
            "efficiency": round(entry["units"] / entry["calls"], 2) if entry["calls"] else 0.0,
        }
    return result


# --- 3. Token cost attribution (approximate) --------------------------------


def _context_tokens(run: Run) -> int:
    """Context size submitted to an LLM call.

    Prefers prompt_tokens (the accumulated context) over total_tokens, which
    also includes completion_tokens and is therefore noisier.
    """
    return run.prompt_tokens or run.total_tokens or 0


def token_cost_attribution(runs: list[Run]) -> dict[str, dict[str, Any]]:
    """Attribute context growth (prompt-token deltas) to tool calls.

    For each LLM step, the positive delta in context size vs the previous LLM
    step is split equally among the tool calls that occurred between them.

    APPROXIMATE: assumes a linear topology ordered by dotted_order, splits
    deltas equally, and ignores compaction (negative deltas). It is a signal
    to guide the Analyst Agent, not an exact measurement.

    Returns {tool_name: {total_delta, calls, avg_delta}} sorted by total_delta.
    """
    sig = significant_runs(runs)
    prev_tokens: int | None = None
    pending_tools: list[str] = []
    attribution: dict[str, dict[str, Any]] = {}

    for run in sig:
        if run.run_type == RunType.TOOL:
            pending_tools.append(run.name)
        elif run.run_type == RunType.LLM:
            tokens = _context_tokens(run)
            if prev_tokens is not None and tokens > prev_tokens and pending_tools:
                per_tool_delta = (tokens - prev_tokens) // len(pending_tools)
                for tname in pending_tools:
                    entry = attribution.setdefault(
                        tname, {"total_delta": 0, "calls": 0, "avg_delta": 0}
                    )
                    entry["total_delta"] += per_tool_delta
                    entry["calls"] += 1
            prev_tokens = tokens
            pending_tools = []

    for entry in attribution.values():
        entry["avg_delta"] = round(entry["total_delta"] / entry["calls"]) if entry["calls"] else 0
    return dict(sorted(attribution.items(), key=lambda x: -x[1]["total_delta"]))


# --- Markdown assembly ------------------------------------------------------


def build_tool_metrics(runs: list[Run]) -> str:
    """Assemble tool_metrics.md from canonical runs."""
    sig = significant_runs(runs)
    if not sig:
        return "# Tool Metrics\n\n(empty trace)\n"

    trace_id = sig[0].trace_id
    sections = [f"# Tool Metrics — trace {trace_id}", ""]

    counts = dict(Counter(r.name for r in _tool_runs(sig)))
    if counts:
        sections += ["## Call frequency", ""]
        sections += [
            f"{name}: {count}" for name, count in sorted(counts.items(), key=lambda x: -x[1])
        ]
        sections.append("")

    repeated = repeated_same_target(runs, _REPEAT_THRESHOLD)
    sections += [f"## Repeated calls on same target (threshold={_REPEAT_THRESHOLD})", ""]
    if repeated:
        for name, targets in repeated.items():
            parts = ", ".join(f"{t} x{c}" for t, c in targets)
            sections.append(f"{name}: {parts}")
        sections.append(
            "Note: repeated calls may be legitimate (e.g. the target changed between "
            "calls). Investigate whether the target changed before concluding redundancy."
        )
    else:
        sections.append("(none above threshold)")
    sections.append("")

    gran = modify_granularity(runs)
    if gran["total_modify_calls"] > 0:
        sections += ["## Modify-call granularity", ""]
        sections.append(
            f"Modify calls: {gran['total_modify_calls']} | "
            f"Units changed: ~{gran['total_units_changed']} | "
            f"Efficiency: {gran['modify_efficiency']} units/call"
        )
        if gran["per_tool"]:
            sections.append("")
            for name, info in gran["per_tool"].items():
                sections.append(
                    f"  {name}: {info['calls']} calls, ~{info['units']} units, "
                    f"{info['efficiency']} units/call"
                )
        sections.append("")

    attrib = token_cost_attribution(runs)
    if attrib:
        sections += ["## Token cost attribution (approximate)", ""]
        for name, info in attrib.items():
            sections.append(
                f"{name}: +{info['total_delta']:,} tokens "
                f"({info['calls']} calls, avg +{info['avg_delta']:,} each)"
            )
        sections.append("")

    return "\n".join(sections) + "\n"
