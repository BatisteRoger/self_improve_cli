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
import os.path
from statistics import median
from typing import Any

from self_improve_cli.domain import Run, RunType
from self_improve_cli.representations import significant_runs

# --- Configurable heuristics ------------------------------------------------

_JUMP_THRESHOLD = 5000
_DROP_RATIO = 0.20
_STALE_TOOL_RESULT_K = 5
_CHARS_PER_TOKEN = 4
_LIFECYCLE_MIN_TOKENS = 250
_LIFECYCLE_DUP_MIN_TOKENS = 50
_LIFECYCLE_TOP_N = 3
# Markdown shows the lifecycle section only when a non-system item re-pays at
# least this many tokens (or duplicate tool outputs exist). JSON is unconditional.
_LIFECYCLE_MD_MIN_REPAID = 1000


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


# --- 2b. Prefix invariance (approximate) -------------------------------------


def _serialize_input(run: Run) -> str:
    """Deterministic serialization of an LLM run's input for prefix comparison.

    One block per message as `role\ntext`, joined by blank lines. Tool-call
    JSON is excluded: the prefix mostly lives in leading messages (system,
    first human), and text alone is enough to detect divergence.
    """
    return "\n\n".join(f"{msg.role}\n{msg.text}" for msg in run.input_messages)


def _common_prefix_chars(a: str, b: str) -> int:
    return len(os.path.commonprefix((a, b)))


def _prefix_invariance(sig: list[Run]) -> list[dict[str, Any]]:
    """Shared input prefix vs earlier main-loop LLM steps (APPROXIMATE).

    For each step i >= 1, computes the *longest* char-level common prefix of
    its serialized input against all earlier steps — adjacent-step comparison
    alone misses re-paid content in multi-node graphs where different nodes
    (each with its own system prompt) interleave on the main loop. Char-level
    (not message-level) so a dynamic field inside the system message (e.g.
    user context) does not collapse the whole prefix to zero.

    Interpretation bounds: shared prefix is the *upper bound* of invariant
    content re-paid each step — constant tool schemas (in `overhead`) add an
    unknown amount. It also approximates provider prefix-cache reuse without
    proving caching occurred.

    Returns one dict per step: {step_index, shared_prefix_chars,
    shared_prefix_tokens, share} — share is None for step 0 (baseline).
    """
    llm_runs = _main_loop_llm_runs(sig)
    if not llm_runs:
        return []
    serialized = [_serialize_input(r) for r in llm_runs]
    results: list[dict[str, Any]] = [
        {"step_index": 0, "shared_prefix_chars": None, "shared_prefix_tokens": None, "share": None}
    ]
    for i in range(1, len(llm_runs)):
        chars = max(_common_prefix_chars(serialized[j], serialized[i]) for j in range(i))
        tokens = _estimate_tokens(serialized[i][:chars])
        prompt = _context_tokens(llm_runs[i])
        results.append(
            {
                "step_index": i,
                "shared_prefix_chars": chars,
                "shared_prefix_tokens": tokens,
                "share": round(tokens / prompt, 2) if prompt else None,
            }
        )
    return results


def prefix_invariance(runs: list[Run]) -> dict[str, Any]:
    """Prefix invariance data: per-step shared prefix + trace aggregate.

    `repaid_tokens` sums shared prefixes across steps — the approximated
    volume of input re-submitted on each main-loop call that had already
    appeared in an earlier step.
    """
    sig = significant_runs(runs)
    per_step = _prefix_invariance(sig)
    shares = [s["share"] for s in per_step if s["share"] is not None]
    repaid = sum(s["shared_prefix_tokens"] or 0 for s in per_step)
    return {
        "steps": per_step,
        "median_share": round(median(shares), 2) if shares else None,
        "repaid_tokens": repaid,
    }


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


# --- 4. Token-cost lifecycle (approximate) -----------------------------------


def _token_lifecycle(
    sig: list[Run],
    min_tokens: int = _LIFECYCLE_MIN_TOKENS,
) -> dict[str, Any]:
    """Track introduced content longevity, repeated exposure, duplication, and churn.

    Operates over main-loop LLM steps (APPROXIMATE, chars/4):
    - `items`: individual messages/tool results with estimated size >= min_tokens,
      ranked by cumulative re-paid tokens across subsequent steps.
    - `duplicate_tool_outputs`: identical tool outputs (>= _LIFECYCLE_DUP_MIN_TOKENS)
      occupying the same step's context under multiple tool calls.
    - `churn_events`: content (>= _LIFECYCLE_DUP_MIN_TOKENS) present at an early step,
      dropped at an intermediate step (e.g. compaction), and re-introduced later.

    Drops (items' `dropped_at_step` and churn) only compare steps of the same
    logical node, identified by the first input message (inferred). A step from
    another graph node interleaved between two steps is not a drop.
    """
    llm_runs = _main_loop_llm_runs(sig)
    if not llm_runs:
        return {
            "items": [],
            "duplicate_tool_outputs": [],
            "churn_events": [],
        }

    name_by_id: dict[str, str] = {}
    for r in llm_runs:
        name_by_id.update(_tool_name_by_call_id(r))

    # Logical node per step: same first input message = same graph node (inferred).
    node_steps: dict[tuple[str, str], list[int]] = {}
    node_of_step: list[tuple[str, str]] = []
    for step_idx, r in enumerate(llm_runs):
        first = r.input_messages[0] if r.input_messages else None
        node = (first.role, first.text) if first else ("", "")
        node_of_step.append(node)
        node_steps.setdefault(node, []).append(step_idx)

    # Track unique message instances: key = (role, tool_call_id_or_idx, text)
    item_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    # Track content-level presence across steps for churn detection: key = (source, text)
    content_steps: dict[tuple[str, str], set[int]] = {}
    # Track duplicate tool outputs within the same step: key = (tool_name, text)
    dup_tracker: dict[tuple[str, str], dict[str, Any]] = {}

    for step_idx, llm_run in enumerate(llm_runs):
        step_tool_contents: dict[tuple[str, str], list[str]] = {}
        non_tool_counts: dict[tuple[str, str], int] = {}

        for msg in llm_run.input_messages:
            est = _estimate_tokens(msg.text)
            if est <= 0:
                continue

            if msg.role == "tool":
                tname = name_by_id.get(msg.tool_call_id or "", msg.tool_call_id or "?")
                source = f"tool:{tname}"
                call_id = msg.tool_call_id or ""
                item_key = ("tool", call_id, msg.text)
                if est >= _LIFECYCLE_DUP_MIN_TOKENS:
                    step_tool_contents.setdefault((tname, msg.text), []).append(call_id)
            else:
                source = msg.role
                call_id = None
                occ = non_tool_counts.get((msg.role, msg.text), 0)
                non_tool_counts[(msg.role, msg.text)] = occ + 1
                item_key = (msg.role, str(occ), msg.text)

            entry = item_meta.get(item_key)
            if entry is None:
                entry = {
                    "role": msg.role,
                    "source": source,
                    "tool_call_id": call_id,
                    "tokens": est,
                    "steps": set(),
                }
                item_meta[item_key] = entry
            entry["steps"].add(step_idx)

            if est >= _LIFECYCLE_DUP_MIN_TOKENS:
                content_steps.setdefault((source, msg.text), set()).add(step_idx)

        for (tname, text), call_ids in step_tool_contents.items():
            if len(call_ids) >= 2:
                est = _estimate_tokens(text)
                existing = dup_tracker.get((tname, text))
                if existing is None:
                    dup_tracker[(tname, text)] = {
                        "tool": tname,
                        "first_step": step_idx,
                        "copies": len(call_ids),
                        "tokens_per_copy": est,
                        "duplicate_tokens": est * (len(call_ids) - 1),
                        "tool_call_ids": [cid for cid in call_ids if cid],
                    }
                elif len(call_ids) > existing["copies"]:
                    existing["copies"] = len(call_ids)
                    existing["duplicate_tokens"] = est * (len(call_ids) - 1)
                    existing["tool_call_ids"] = [cid for cid in call_ids if cid]

    items: list[dict[str, Any]] = []
    for entry in item_meta.values():
        est = entry["tokens"]
        if est < min_tokens:
            continue
        steps_sorted = sorted(entry["steps"])
        intro = steps_sorted[0]
        last = steps_sorted[-1]
        count = len(steps_sorted)
        steps_set = entry["steps"]
        dropped_at = next(
            (s for s in node_steps[node_of_step[intro]] if s > intro and s not in steps_set),
            None,
        )
        items.append(
            {
                "role": entry["role"],
                "source": entry["source"],
                "tool_call_id": entry["tool_call_id"],
                "introduced_step": intro,
                "last_step": last,
                "steps_present": count,
                "dropped_at_step": dropped_at,
                "tokens": est,
                "cumulative_tokens": est * count,
                "repaid_tokens": est * (count - 1),
            }
        )

    items.sort(key=lambda x: (-x["repaid_tokens"], -x["cumulative_tokens"], x["introduced_step"]))

    churn_events: list[dict[str, Any]] = []
    for (source, text), steps_set in content_steps.items():
        for positions in node_steps.values():
            present = [i for i, s in enumerate(positions) if s in steps_set]
            gap = next(
                ((a, b) for a, b in zip(present, present[1:], strict=False) if b > a + 1),
                None,
            )
            if gap is None:
                continue
            churn_events.append(
                {
                    "source": source,
                    "tokens": _estimate_tokens(text),
                    "introduced_step": positions[present[0]],
                    "dropped_at_step": positions[gap[0] + 1],
                    "reintroduced_at_step": positions[gap[1]],
                }
            )
            break

    churn_events.sort(key=lambda x: (-x["tokens"], x["dropped_at_step"]))
    duplicates = sorted(dup_tracker.values(), key=lambda x: -x["duplicate_tokens"])

    return {
        "items": items,
        "duplicate_tool_outputs": duplicates,
        "churn_events": churn_events,
    }


def token_lifecycle(
    runs: list[Run],
    min_tokens: int = _LIFECYCLE_MIN_TOKENS,
) -> dict[str, Any]:
    """Token-cost lifecycle over main-loop LLM steps (approximate).

    Tracks where large context items were introduced, how many subsequent steps
    re-submitted them (cumulative and re-paid token cost), whether identical tool
    outputs occupied context simultaneously, and whether dropped content was
    re-introduced later (context churn).
    """
    return _token_lifecycle(significant_runs(runs), min_tokens=min_tokens)


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

        # 1b. Prefix invariance
        invariance = _prefix_invariance(sig)
        if len(invariance) > 1:
            sections += ["## Static context (prefix invariance)", ""]
            for s in invariance[1:]:
                share = f"{int(s['share'] * 100)}%" if s["share"] is not None else "n/a"
                sections.append(
                    f"Step {s['step_index']:3d} | shared_prefix: "
                    f"{_fmt_k(s['shared_prefix_tokens'])} ({share} of "
                    f"{_context_tokens(llm_runs[s['step_index']]):,} prompt tokens)"
                )
            shares = [s["share"] for s in invariance[1:] if s["share"] is not None]
            repaid = sum(s["shared_prefix_tokens"] or 0 for s in invariance)
            median_str = f"{int(median(shares) * 100)}%" if shares else "n/a"
            sections += [
                "",
                f"Median share: {median_str} | "
                f"~{_fmt_k(repaid)} tokens re-paid across {len(invariance) - 1} steps",
                "",
                "Note: APPROXIMATE — longest char-level common prefix between this step's "
                "input and any earlier step's, chars/4. Upper bound of content re-paid per "
                "step; constant tool schemas add an unknown amount. Approximates provider "
                "prefix-cache reuse — does not prove caching occurred.",
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

    # 3. Token-cost lifecycle
    # Bounded: shown only above a re-paid threshold; churn and the full lists
    # stay in --format json.
    lifecycle = _token_lifecycle(sig)
    md_items = [i for i in lifecycle["items"] if i["role"] != "system" and i["repaid_tokens"] > 0]
    dups = lifecycle["duplicate_tool_outputs"]
    if dups or any(i["repaid_tokens"] >= _LIFECYCLE_MD_MIN_REPAID for i in md_items):
        sections += ["## Token-cost lifecycle (approximate)", ""]
        for item in md_items[:_LIFECYCLE_TOP_N]:
            call_ref = f" ({item['tool_call_id']})" if item["tool_call_id"] else ""
            drop_ref = (
                f" [dropped at step {item['dropped_at_step']}]"
                if item["dropped_at_step"] is not None
                else ""
            )
            sections.append(
                f"Steps {item['introduced_step']}-{item['last_step']} | "
                f"{item['source']}{call_ref} | ~{_fmt_k(item['tokens'])} x "
                f"{item['steps_present']} steps, ~{_fmt_k(item['repaid_tokens'])} re-paid"
                f"{drop_ref}"
            )
        extra_items = len(md_items) - _LIFECYCLE_TOP_N
        if extra_items > 0:
            sections.append(f"... ({extra_items} more items, see --format json)")

        for dup in dups[:_LIFECYCLE_TOP_N]:
            cids = f" ({', '.join(dup['tool_call_ids'])})" if dup["tool_call_ids"] else ""
            sections.append(
                f"Duplicate tool output at step {dup['first_step']}: "
                f"{dup['tool']} x{dup['copies']}{cids} "
                f"(~{_fmt_k(dup['duplicate_tokens'])} redundant tokens)"
            )
        if len(dups) > _LIFECYCLE_TOP_N:
            sections.append(
                f"... ({len(dups) - _LIFECYCLE_TOP_N} more duplicates, see --format json)"
            )

        sections += [
            "",
            "Note: APPROXIMATE (chars/4), ignores prefix-caching; not proof of unused content.",
            "",
        ]

    return "\n".join(sections) + "\n"


def context_metrics_data(runs: list[Run]) -> dict[str, Any]:
    """Structured context metrics for JSON output (composable contract).

    Returns a dict with trace_id, token decomposition per step, stale ratios,
    and the growth curve — the same data as build_context_metrics, structured.
    """
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": runs[0].trace_id if runs else "", "empty": True}

    trace_id = sig[0].trace_id
    llm_runs = _main_loop_llm_runs(sig)
    decomps = [_token_decomposition(r) for r in llm_runs]
    stale_ratios = _stale_tool_result_ratio(sig)
    curve = _growth_curve(sig)

    steps: list[dict[str, Any]] = []
    for i, (decomp, stale) in enumerate(zip(decomps, stale_ratios)):
        steps.append(
            {
                "step": i,
                "prompt_tokens": decomp["prompt_tokens"],
                "categories": decomp["categories"],
                "per_tool": decomp["per_tool"],
                "overhead": decomp["overhead"],
                "stale_tool_result_ratio": stale,
            }
        )

    return {
        "trace_id": trace_id,
        "token_decomposition": steps,
        "prefix_invariance": prefix_invariance(runs),
        "growth_curve": curve,
        "token_lifecycle": _token_lifecycle(sig),
        "notes": {
            "token_categories": "chars/4 estimates (APPROXIMATE)",
            "overhead": "Residual vs real prompt_tokens (tool schemas + estimation error)",
            "stale_ratio": (
                f"% of tool results older than the {_STALE_TOOL_RESULT_K} most recent. "
                "Measures age, not usefulness."
            ),
            "growth_curve": "Nested-LLM runs (tools spawning their own LLM) are excluded.",
            "prefix_invariance": (
                "Longest char-level common prefix between a step's input and any earlier "
                "step's, chars/4 (APPROXIMATE). Upper bound of content re-paid per step."
            ),
            "token_lifecycle": (
                "Payload retention across main-loop steps, chars/4 (APPROXIMATE). "
                "Cumulative tokens do not account for provider prefix-caching; long "
                "retention signals potentially avoidable context, not unused content. "
                "Drops and churn compare only steps of the same logical node (same "
                "first input message, inferred)."
            ),
        },
    }
