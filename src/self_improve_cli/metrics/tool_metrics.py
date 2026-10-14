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
from datetime import datetime
from typing import Any

from self_improve_cli.domain import Run, RunType
from self_improve_cli.representations import significant_runs
from self_improve_cli.representations.common import _is_infra_error, _tool_result_text

# --- Configurable heuristics ------------------------------------------------

_TARGET_KEY_CANDIDATES = ("path", "file", "filename", "file_path", "slug", "target", "uri", "url")
_TARGET_KEY_IGNORE = ("type", "run_id", "id", "tool_call_id")
_TARGET_FALLBACK_MAX_LEN = 100
_MODIFY_NAME_PATTERN = ("edit", "write", "patch", "update", "replace", "set")
_MODIFY_ARG_HINTS = ("old_string", "new_string", "content", "patch")
_REPEAT_THRESHOLD = 3
_OVERSIZED_TOOL_TOKENS = 2000
_CHARS_PER_TOKEN = 4


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


# --- 4. Latency decomposition -----------------------------------------------


def _parse_interval(run: Run) -> tuple[float, float] | None:
    """Return (start_epoch_s, end_epoch_s) for a run if valid timestamps exist."""
    if not (run.start_time and run.end_time):
        return None
    try:
        start = datetime.fromisoformat(str(run.start_time).replace("Z", "+00:00")).timestamp()
        end = datetime.fromisoformat(str(run.end_time).replace("Z", "+00:00")).timestamp()
        if end < start:
            return None
        return (start, end)
    except (ValueError, TypeError):
        return None


def _merged_duration(intervals: list[tuple[float, float]]) -> float:
    """Sum interval durations after merging overlaps (parallel runs counted once)."""
    if not intervals:
        return 0.0
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    cur_start, cur_end = sorted_iv[0]
    total = 0.0
    for s, e in sorted_iv[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
    total += cur_end - cur_start
    return total


def _is_agent_error(run: Run) -> bool:
    """True when a run errored and the error is not an infra cancellation."""
    if run.error:
        return not _is_infra_error(str(run.error))
    return run.status == "error"


def latency_decomposition(runs: list[Run]) -> dict[str, Any]:
    """Decompose wall-clock latency into main-loop LLM time, tool time, and gaps.

    Overlapping intervals (e.g. parallel tool calls) are merged so concurrent
    work is not double-counted as sequential. Nested LLM calls inside tool
    subagents are already covered by their parent tool interval.
    """
    sig = significant_runs(runs)
    if not sig:
        return {"available": False}

    tool_ids = {r.id for r in sig if r.run_type == RunType.TOOL}
    llm_intervals: list[tuple[float, float]] = []
    tool_intervals: list[tuple[float, float]] = []
    failed_tool_intervals: list[tuple[float, float]] = []
    all_intervals: list[tuple[float, float]] = []
    per_tool: dict[str, dict[str, Any]] = {}

    for run in sig:
        iv = _parse_interval(run)
        if iv is None:
            continue
        all_intervals.append(iv)
        dur = iv[1] - iv[0]
        if run.run_type == RunType.LLM and run.parent_run_id not in tool_ids:
            llm_intervals.append(iv)
        elif run.run_type == RunType.TOOL:
            tool_intervals.append(iv)
            if _is_agent_error(run):
                failed_tool_intervals.append(iv)
            t_entry = per_tool.setdefault(run.name, {"calls": 0, "total_s": 0.0, "max_s": 0.0})
            t_entry["calls"] += 1
            t_entry["total_s"] += dur
            t_entry["max_s"] = max(t_entry["max_s"], dur)

    if not (llm_intervals or tool_intervals):
        return {"available": False}

    trace_start = min(s for s, _ in all_intervals)
    trace_end = max(e for _, e in all_intervals)
    total_s = max(trace_end - trace_start, 0.0)
    llm_s = _merged_duration(llm_intervals)
    tool_s = _merged_duration(tool_intervals)
    failed_tool_s = _merged_duration(failed_tool_intervals)
    active_s = _merged_duration(llm_intervals + tool_intervals)
    unexplained_s = max(total_s - active_s, 0.0)

    per_tool_out = {
        name: {
            "calls": info["calls"],
            "total_s": round(info["total_s"], 2),
            "max_s": round(info["max_s"], 2),
        }
        for name, info in sorted(per_tool.items(), key=lambda x: -x[1]["total_s"])
    }

    return {
        "available": True,
        "total_s": round(total_s, 2),
        "llm_time_s": round(llm_s, 2),
        "tool_time_s": round(tool_s, 2),
        "failed_tool_time_s": round(failed_tool_s, 2),
        "unexplained_gap_s": round(unexplained_s, 2),
        "per_tool": per_tool_out,
    }


# --- 5. Process & harness friction signals ----------------------------------


def _tool_signature_label(run: Run) -> tuple[str, str]:
    """Return (canonical_key, display_label) for a tool run."""
    args = _tool_args(run)
    target = extract_target_key(run.name, args)
    if target:
        return (f"{run.name}:{target}", f"{run.name}({target})")
    raw = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    return (f"{run.name}:{raw}", run.name)


def process_and_friction_signals(
    runs: list[Run],
    oversized_tokens: int = _OVERSIZED_TOOL_TOKENS,
) -> dict[str, Any]:
    """Surface deterministic tool/process friction signals with run_id citations.

    Detectors:
    - `failed_retries`: a failed tool call followed by another call to the same
      tool — distinguishes `identical_args` (maps to `recovery.failed_recovery`)
      from `adapted_after_error` (maps to `tool.probabilistic_discovery`).
    - `oscillations`: A -> B -> A alternation across consecutive tool calls
      (maps to `control.nonprogress_loop`).
    - `oversized_outputs`: tool outputs >= oversized_tokens (chars/4,
      candidate `harness.observation_pack`).
    """
    sig = significant_runs(runs)
    tools = _tool_runs(sig)

    # 1. Failed-command retries (by tool name sequence)
    failed_retries: list[dict[str, Any]] = []
    last_failed_by_tool: dict[str, Run] = {}
    for run in tools:
        prev_failed = last_failed_by_tool.get(run.name)
        if prev_failed is not None:
            prev_args = _tool_args(prev_failed)
            cur_args = _tool_args(run)
            target = extract_target_key(run.name, cur_args) or extract_target_key(
                prev_failed.name, prev_args
            )
            if cur_args == prev_args:
                failed_retries.append(
                    {
                        "tool": run.name,
                        "target": target,
                        "failed_run_id": prev_failed.id,
                        "retry_run_id": run.id,
                        "mode": "identical_args",
                        "suggested_pattern": "recovery.failed_recovery",
                    }
                )
            elif not _is_agent_error(run):
                failed_retries.append(
                    {
                        "tool": run.name,
                        "target": target,
                        "failed_run_id": prev_failed.id,
                        "retry_run_id": run.id,
                        "mode": "adapted_after_error",
                        "suggested_pattern": "tool.probabilistic_discovery",
                    }
                )

        if _is_agent_error(run):
            last_failed_by_tool[run.name] = run
        else:
            last_failed_by_tool.pop(run.name, None)

    # 2. Oscillation: contiguous A -> B -> A (-> B ...) alternation
    oscillations: list[dict[str, Any]] = []
    labels = [_tool_signature_label(r) for r in tools]
    i = 0
    while i + 2 < len(tools):
        key_a, disp_a = labels[i]
        key_b, disp_b = labels[i + 1]
        key_c, _ = labels[i + 2]
        if key_a == key_c and key_a != key_b:
            end = i + 2
            while end + 1 < len(tools):
                expected = key_b if (end - i) % 2 == 0 else key_a
                if labels[end + 1][0] == expected:
                    end += 1
                else:
                    break
            osc_runs = tools[i : end + 1]
            oscillations.append(
                {
                    "sequence": [disp_a, disp_b, disp_a],
                    "run_ids": [r.id for r in osc_runs],
                    "length": len(osc_runs),
                    "suggested_pattern": "control.nonprogress_loop",
                }
            )
            i = end + 1
        else:
            i += 1

    # 3. Oversized tool outputs
    oversized_outputs: list[dict[str, Any]] = []
    for run in tools:
        res_text = _tool_result_text(run)
        est_tokens = len(res_text) // _CHARS_PER_TOKEN
        if est_tokens >= oversized_tokens:
            oversized_outputs.append(
                {
                    "run_id": run.id,
                    "tool": run.name,
                    "target": extract_target_key(run.name, _tool_args(run)),
                    "estimated_tokens": est_tokens,
                    "suggested_candidate": "harness.observation_pack",
                }
            )
    oversized_outputs.sort(key=lambda x: -x["estimated_tokens"])

    return {
        "failed_retries": failed_retries,
        "oscillations": oscillations,
        "oversized_outputs": oversized_outputs,
    }


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

    lat = latency_decomposition(runs)
    if lat.get("available"):
        sections += ["## Latency decomposition", ""]
        sections.append(
            f"Total: {lat['total_s']:.2f}s | "
            f"Main-loop LLM: {lat['llm_time_s']:.2f}s | "
            f"Tools (merged): {lat['tool_time_s']:.2f}s | "
            f"Failed tools: {lat['failed_tool_time_s']:.2f}s | "
            f"Unexplained gap: {lat['unexplained_gap_s']:.2f}s"
        )
        if lat["per_tool"]:
            parts = [
                f"{name}={info['total_s']:.2f}s ({info['calls']}x, max {info['max_s']:.2f}s)"
                for name, info in lat["per_tool"].items()
            ]
            sections.append("Per tool: " + ", ".join(parts))
        sections.append(
            "Note: overlapping parallel intervals are merged so concurrent tool calls "
            "are not double-counted."
        )
        sections.append("")

    signals = process_and_friction_signals(runs)
    if signals["failed_retries"] or signals["oscillations"] or signals["oversized_outputs"]:
        sections += ["## Process & friction signals", ""]
        for fr in signals["failed_retries"]:
            t_str = f" ({fr['target']})" if fr["target"] else ""
            sections.append(
                f"Failed retry [{fr['mode']}]: {fr['tool']}{t_str} "
                f"{fr['failed_run_id']} -> {fr['retry_run_id']} "
                f"(candidate pattern: {fr['suggested_pattern']})"
            )
        for osc in signals["oscillations"]:
            seq_str = " -> ".join(osc["sequence"])
            rids = ", ".join(osc["run_ids"])
            sections.append(
                f"Oscillation ({osc['length']} calls): {seq_str} "
                f"[{rids}] (candidate pattern: {osc['suggested_pattern']})"
            )
        for ov in signals["oversized_outputs"]:
            t_str = f" ({ov['target']})" if ov["target"] else ""
            sections.append(
                f"Oversized tool output: {ov['tool']}{t_str} id={ov['run_id']} "
                f"~{ov['estimated_tokens']:,} tokens "
                f"(candidate: {ov['suggested_candidate']})"
            )
        sections.append(
            "Note: deterministic signals point to candidate patterns — inspect the cited "
            "run IDs before recording a finding."
        )
        sections.append("")

    return "\n".join(sections) + "\n"


def tool_metrics_data(runs: list[Run]) -> dict[str, Any]:
    """Structured tool metrics for JSON output (composable contract).

    Returns a dict with trace_id and the same sections as build_tool_metrics,
    but as structured data instead of prose.
    """
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": runs[0].trace_id if runs else "", "empty": True}

    trace_id = sig[0].trace_id
    counts = dict(Counter(r.name for r in _tool_runs(sig)))
    repeated = repeated_same_target(runs, _REPEAT_THRESHOLD)
    gran = modify_granularity(runs)
    attrib = token_cost_attribution(runs)
    lat = latency_decomposition(runs)
    signals = process_and_friction_signals(runs)

    return {
        "trace_id": trace_id,
        "call_frequency": counts,
        "repeated_same_target": {
            name: [{"target": t, "count": c} for t, c in targets]
            for name, targets in repeated.items()
        },
        "modify_granularity": gran if gran["total_modify_calls"] > 0 else None,
        "token_cost_attribution": attrib if attrib else None,
        "latency_decomposition": lat if lat.get("available") else None,
        "process_and_friction_signals": signals,
        "notes": {
            "repeated_calls": (
                "Repeated calls may be legitimate (e.g. the target changed "
                "between calls). Investigate whether the target changed before "
                "concluding redundancy."
            ),
            "token_attribution": (
                "Approximate — assumes the tool is the only cause of context growth."
            ),
            "latency_decomposition": (
                "Overlapping parallel intervals are merged so concurrent calls are "
                "not double-counted."
            ),
            "process_signals": (
                "Deterministic candidate signals — verify cited run IDs before "
                "confirming a finding."
            ),
        },
    }
