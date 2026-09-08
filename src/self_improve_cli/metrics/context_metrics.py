"""Context metrics — deterministic token decomposition & growth analysis.

Pure functions over canonical Run objects. No SDK dependency, no LLM calls:
deterministic and recomputable from sanitized data.

The metrics are raw facts (token counts per category, deltas, stale
tool-result ratio). Interpreting them ("context bloat", "architecture
issue") is the job of the Analyst Agent that consumes this file, not of
this module.

APPROXIMATE: token categories use chars/4 estimates; overhead is a residual;
stale tool-result ratio is a position-based proxy. See notes in each section.
"""

from __future__ import annotations

import json
from typing import Any

from self_improve_cli.domain import Run, RunType
from self_improve_cli.representations import significant_runs

# --- Configurable heuristics ------------------------------------------------

_JUMP_THRESHOLD = 5000
_DROP_RATIO = 0.20
_STALE_TOOL_RESULT_K = 5
_CHARS_PER_TOKEN = 4


# --- Helpers ----------------------------------------------------------------


def _context_tokens(run: Run) -> int:
    """Context size submitted to an LLM call.

    Prefers prompt_tokens (the accumulated context) over total_tokens, which
    also includes completion_tokens and is therefore noisier.
    """
    return run.prompt_tokens or run.total_tokens or 0


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4. APPROXIMATE."""
    return len(text) // _CHARS_PER_TOKEN


def _tool_name_by_call_id(run: Run) -> dict[str, str]:
    """Map tool_call_id -> tool name from AIMessages' tool_calls in a step."""
    mapping: dict[str, str] = {}
    for msg in run.input_messages:
        if msg.role == "ai":
            for tc in msg.tool_calls:
                if tc.id:
                    mapping[tc.id] = tc.name
    return mapping


def _main_loop_llm_runs(sig: list[Run]) -> list[Run]:
    """LLM runs that belong to the agent's main loop.

    Excludes LLM runs whose parent is a tool run (e.g., a critic subagent
    calling ChatAnthropic internally). Those nested LLMs have their own
    token counts that would pollute the growth curve with false drops/jumps.
    """
    tool_ids = {r.id for r in sig if r.run_type == RunType.TOOL}
    return [r for r in sig if r.run_type == RunType.LLM and r.parent_run_id not in tool_ids]


# --- 1. Token decomposition -------------------------------------------------


def _token_decomposition(llm_run: Run) -> dict[str, Any]:
    """Decompose the context of a single LLM step into token categories.

    Categories (estimated via chars/4): system, human, ai (text + tool_calls
    JSON), tool. overhead is the residual against the real prompt_tokens and
    absorbs tool schemas/docstrings plus estimation error (APPROXIMATE).

    Tool-result tokens are attributed per tool name via tool_call_id.

    Returns {categories, per_tool, overhead, prompt_tokens}.
    """
    msgs = llm_run.input_messages
    categories: dict[str, int] = {"system": 0, "human": 0, "ai": 0, "tool": 0}
    per_tool: dict[str, int] = {}
    name_by_id = _tool_name_by_call_id(llm_run)

    for msg in msgs:
        role = msg.role
        est = _estimate_tokens(msg.text)
        if role == "system":
            categories["system"] += est
        elif role == "human":
            categories["human"] += est
        elif role == "ai":
            categories["ai"] += est
            if msg.tool_calls:
                categories["ai"] += _estimate_tokens(
                    json.dumps([tc.name for tc in msg.tool_calls], default=str)
                )
        elif role == "tool":
            categories["tool"] += est
            name = name_by_id.get(msg.tool_call_id or "", msg.tool_call_id or "?")
            per_tool[name] = per_tool.get(name, 0) + est

    real_total = _context_tokens(llm_run)
    overhead = max(real_total - sum(categories.values()), 0)

    return {
        "categories": categories,
        "per_tool": per_tool,
        "overhead": overhead,
        "prompt_tokens": real_total,
    }


def token_decomposition(runs: list[Run]) -> list[dict[str, Any]]:
    """Decompose context tokens per main-loop LLM step.

    Returns a list of dicts (one per main-loop LLM step) with categories,
    per-tool sub-totals, overhead, and prompt_tokens.
    """
    sig = significant_runs(runs)
    return [_token_decomposition(r) for r in _main_loop_llm_runs(sig)]


# --- 2. Stale tool-result ratio (approximate) --------------------------------


def _stale_tool_result_ratio(sig: list[Run], k: int = _STALE_TOOL_RESULT_K) -> list[dict[str, Any]]:
    """Stale tool-result ratio per main-loop LLM step (APPROXIMATE).

    Within each step's accumulated context, a ToolMessage is "recent" if it is
    among the K most recent tool results; older ToolMessages count as "stale"
    (old tool output still occupying context). This is a deterministic,
    position-based proxy: it does not know whether old content is actually
    reused, only that it is old. A high ratio does not mean the old results
    are useless — it means they are old.

    Returns one dict per step: {step_index, total_tool_msgs, stale_tool_msgs, ratio}.
    """
    results: list[dict[str, Any]] = []
    for i, llm_run in enumerate(_main_loop_llm_runs(sig)):
        tool_msgs = [m for m in llm_run.input_messages if m.role == "tool"]
        total = len(tool_msgs)
        stale = max(total - k, 0)
        results.append(
            {
                "step_index": i,
                "total_tool_msgs": total,
                "stale_tool_msgs": stale,
                "ratio": round(stale / total, 2) if total else 0.0,
            }
        )
    return results


def stale_tool_result_ratio(runs: list[Run], k: int = _STALE_TOOL_RESULT_K) -> list[dict[str, Any]]:
    """Stale tool-result ratio at each main-loop LLM step (approximate).

    Measures age, not usefulness. Older tool results may still be relevant.
    """
    return _stale_tool_result_ratio(significant_runs(runs), k)


# Backward-compat alias. The previous name was misleading: "dead context"
# implied the content was not contributing, but the metric only measures age.
dead_context_ratio = stale_tool_result_ratio


# --- 3. Growth curve --------------------------------------------------------


def _growth_curve(sig: list[Run]) -> dict[str, Any]:
    """Compute growth curve data over main-loop LLM steps.

    Returns dict with:
    - steps: list of {from_step, to_step, delta, is_jump, is_drop}
    - has_drops: bool
    - is_monotonic: bool (no drops detected)
    """
    llm_runs = _main_loop_llm_runs(sig)
    tokens = [_context_tokens(r) for r in llm_runs]

    steps: list[dict[str, Any]] = []
    has_drops = False

    for i in range(1, len(tokens)):
        delta = tokens[i] - tokens[i - 1]
        is_jump = delta > _JUMP_THRESHOLD
        is_drop = delta < -int(_DROP_RATIO * tokens[i - 1])
        if is_drop:
            has_drops = True
        steps.append(
            {
                "from_step": i - 1,
                "to_step": i,
                "delta": delta,
                "is_jump": is_jump,
                "is_drop": is_drop,
            }
        )

    return {
        "steps": steps,
        "has_drops": has_drops,
        "is_monotonic": not has_drops,
    }


def growth_curve(runs: list[Run]) -> dict[str, Any]:
    """Growth curve data: deltas, jumps, drops, monotonicity.

    Computed over main-loop LLM steps (nested LLMs excluded).
    """
    return _growth_curve(significant_runs(runs))


# --- Markdown assembly ------------------------------------------------------


def _fmt_delta(n: int) -> str:
    """Format a signed token delta: 3842 -> '+3,842', -1200 -> '-1,200'."""
    return f"{n:+,}"


def _fmt_k(n: int) -> str:
    """Format token count in K: 6000 -> '6K', 500 -> '0.5K'."""
    if n >= 1000:
        return f"{n / 1000:.0f}K"
    return f"{n / 1000:.1f}K"


def build_context_metrics(runs: list[Run]) -> str:
    """Assemble context_metrics.md from canonical runs."""
    sig = significant_runs(runs)
    if not sig:
        return "# Context Metrics\n\n(empty trace)\n"

    trace_id = sig[0].trace_id
    llm_runs = _main_loop_llm_runs(sig)
    sections = [f"# Context Metrics — trace {trace_id}", ""]

    # 1. Token decomposition
    if llm_runs:
        sections += ["## Token decomposition (per main-loop LLM step)", ""]
        decomps = [_token_decomposition(r) for r in llm_runs]
        stale_ratios = _stale_tool_result_ratio(sig)

        for i, (decomp, stale) in enumerate(zip(decomps, stale_ratios)):
            cats = decomp["categories"]
            pt = decomp["prompt_tokens"]
            per_tool = decomp["per_tool"]
            oh = decomp["overhead"]

            tool_str = ""
            if per_tool:
                tool_parts = ", ".join(
                    f"{k}={_fmt_k(v)}" for k, v in sorted(per_tool.items(), key=lambda x: -x[1])
                )
                tool_str = f" ({tool_parts})"

            sections.append(
                f"Step {i:3d} | {pt:,} | "
                f"sys:{_fmt_k(cats['system'])} human:{_fmt_k(cats['human'])} "
                f"ai:{_fmt_k(cats['ai'])} tool:{_fmt_k(cats['tool'])}{tool_str} "
                f"overhead:{_fmt_k(oh)} | stale: {int(stale['ratio'] * 100)}%"
            )
        sections += [
            "",
            "Notes: token categories are chars/4 estimates (APPROXIMATE); overhead is "
            "the residual vs real prompt_tokens (tool schemas + estimation error). "
            f"stale = % of tool results older than the {_STALE_TOOL_RESULT_K} most recent. "
            "This measures age, not usefulness — older results may still be relevant.",
            "",
        ]

    # 2. Growth curve
    curve = _growth_curve(sig)
    if curve["steps"]:
        sections += ["## Growth curve", ""]
        parts = []
        for s in curve["steps"]:
            label = ""
            if s["is_jump"]:
                label = " (jump)"
            elif s["is_drop"]:
                label = " (drop)"
            parts.append(f"{s['from_step']}->{s['to_step']} {_fmt_delta(s['delta'])}{label}")

        line = " | ".join(parts)
        sections.append(f"Steps: {line}")
        sections.append(f"Drops: {'yes' if curve['has_drops'] else 'none'}")
        sections.append(f"Monotonic growth: {'yes' if curve['is_monotonic'] else 'no'}")
        sections.append("")
        sections.append(
            "Note: nested-LLM runs (tools spawning their own LLM) are excluded from this curve."
        )
        sections.append("")

    return "\n".join(sections) + "\n"
