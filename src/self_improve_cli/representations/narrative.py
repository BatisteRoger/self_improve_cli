"""L2 narrative — chronological story with message deltas per LLM call."""

from __future__ import annotations

import json
from typing import Any

from self_improve_cli.domain import Message, Run, RunType
from self_improve_cli.representations.common import (
    _MAX_MSG_CHARS,
    _MAX_TOOL_CHARS,
    _format_error,
    _format_tool_calls,
    _latency_s,
    _message_signature,
    _truncate,
    significant_runs,
    tool_result_text,
)


def _narrative_llm_step_full(run: Run, previous_msgs: list[Message]) -> tuple[str, list[Message]]:
    """Render one LLM call as a delta against the previous call's messages.

    Uses a common-prefix walk: messages are compared in order, and the walk
    stops at the first mismatch. Everything after the mismatch is printed in full.
    This is the original behavior — useful for humans skimming start-to-end,
    but degrades when the system message changes every step (dynamic prompts).
    """
    msgs = run.input_messages
    prev_sigs = [_message_signature(m) for m in previous_msgs]
    sigs = [_message_signature(m) for m in msgs]
    common = 0
    while common < min(len(prev_sigs), len(sigs)) and prev_sigs[common] == sigs[common]:
        common += 1

    lines = []
    if common:
        lines.append(f"(context: {common} unchanged messages, {len(msgs) - common} new)")
    for msg in msgs[common:]:
        lines.append(f"[{msg.role}] {_truncate(msg.text, _MAX_MSG_CHARS)}")
        if msg.tool_calls:
            lines.append(_format_tool_calls(msg.tool_calls))

    out = run.output_message
    lines.append("=> response:")
    if out is None:
        lines.append(
            f"(no parsed output) {_truncate(json.dumps(run.outputs, default=str), _MAX_TOOL_CHARS)}"
        )
    else:
        if out.text:
            lines.append(f"[{out.role}] {_truncate(out.text, _MAX_MSG_CHARS)}")
        if out.tool_calls:
            lines.append(_format_tool_calls(out.tool_calls))
    return "\n".join(lines), msgs


def _narrative_llm_step_compact(
    run: Run, previous_msgs: list[Message], step_num: int
) -> tuple[str, list[Message]]:
    """Render one LLM call as a per-index delta against the previous call's messages.

    For each message at index i, compare its signature against the message at
    index i in the previous step independently (no early stop on first mismatch).
    Identical messages collapse to one line; only genuinely new/changed messages
    are printed in full. This fixes the cascade where a dynamic system prompt
    causes the entire message list to be re-dumped every step.
    """
    msgs = run.input_messages
    prev_sigs = [_message_signature(m) for m in previous_msgs]
    sigs = [_message_signature(m) for m in msgs]

    lines = []
    unchanged = 0
    for i, msg in enumerate(msgs):
        if i < len(prev_sigs) and prev_sigs[i] == sigs[i]:
            unchanged += 1
            continue
        if unchanged:
            lines.append(f"({unchanged} unchanged messages)")
            unchanged = 0
        lines.append(f"[{msg.role}] {_truncate(msg.text, _MAX_MSG_CHARS)}")
        if msg.tool_calls:
            lines.append(_format_tool_calls(msg.tool_calls))
    if unchanged:
        lines.append(f"({unchanged} unchanged messages)")

    out = run.output_message
    lines.append("=> response:")
    if out is None:
        lines.append(
            f"(no parsed output) {_truncate(json.dumps(run.outputs, default=str), _MAX_TOOL_CHARS)}"
        )
    else:
        if out.text:
            lines.append(f"[{out.role}] {_truncate(out.text, _MAX_MSG_CHARS)}")
        if out.tool_calls:
            lines.append(_format_tool_calls(out.tool_calls))
    return "\n".join(lines), msgs


_CHARS_PER_TOKEN = 4
_DROP_RATIO = 0.20
_MIN_PAYLOAD_CHARS = 80
# Canonical per-step framing overhead: the largest of the markdown header
# line or the JSON step metadata. One canonical cost drives the shared
# budget partition so Markdown and JSON select identical step sets.
_BUDGET_JSON_META_CHARS = 220
# Canonical base overhead (header lines, task framing, budget/range notes,
# JSON wrapper keys) charged before any step payload.
_BUDGET_BASE_OVERHEAD = 400
# Per-omitted-step overhead charged against the budget so that markdown
# markers and JSON omitted_steps entries are counted, not free.
_BUDGET_OMITTED_STEP_CHARS = 120


def _step_priority(
    run: Run,
    step: int,
    *,
    first_step: int,
    last_llm_step: int,
    tool_ids: set[str],
    is_compaction_step: bool,
) -> int:
    """Priority tier for `--budget` selection (lower = higher priority).

    Tier 1: anchor steps (first step, final LLM reply), errors, subagent runs
            (parent is a tool run), and compaction boundaries.
    Tier 2: intermediate main-loop LLM reasoning and tool-call decisions.
    Tier 3: routine tool results and chain node outputs.
    """
    has_error = bool(run.error) or run.status not in ("success", "")
    is_subagent = bool(run.parent_run_id and run.parent_run_id in tool_ids)
    if (
        step == first_step
        or step == last_llm_step
        or has_error
        or is_subagent
        or is_compaction_step
    ):
        return 1
    if run.run_type == RunType.LLM:
        return 2
    return 3


def _format_omitted_marker(omitted: list[dict[str, Any]], compact: bool = False) -> str:
    """Format a contiguous block of budget-omitted steps with run_id references.

    `compact` drops the per-step previews — the partition picks this form when
    the budget is too tight for full markers, so the marker itself stays
    within budget instead of evicting an anchor step to pay for its own text.
    """
    count = len(omitted)
    est_tokens = sum(item["estimated_tokens"] for item in omitted)
    first_s = omitted[0]["step"]
    last_s = omitted[-1]["step"]
    range_hint = f"--from {first_s} --to {last_s}"
    if compact:
        return (
            f"[omitted {count} step(s), ~{est_tokens} tokens "
            f"— inspect with `narrative {range_hint}` or `run-detail`]"
        )
    previews = [
        f"step {item['step']} {item['type']} {item['name']} (id={item['run_id']})"
        for item in omitted[:3]
    ]
    if count > 3:
        previews.append(f"+{count - 3} more")
    return (
        f"[omitted {count} step(s), ~{est_tokens} tokens: "
        + ", ".join(previews)
        + f" — inspect with `narrative {range_hint}` or `run-detail`]"
    )


def _collect_narrative_steps(
    sig: list[Run],
    mode: str,
    step_from: int | None,
    step_to: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Walk significant runs once; build the renderable step list.

    Returns (items, total_steps). Each item carries everything both renderers
    need: markdown uses "header"/"body"; JSON uses "type", "run_id", "name",
    "tokens", "latency_s", "status", "error" plus the payload fields ("body"
    for llm, "args"/"result" for tool, "output" for node). "priority",
    "framing" (canonical per-step overhead) and "estimated_tokens" are
    precomputed for the shared --budget partition.
    """
    tool_ids = {r.id for r in sig if r.run_type == RunType.TOOL}
    last_llm_step = 0
    for idx, r in enumerate(sig[1:], start=1):
        if r.run_type == RunType.LLM and not _step_out_of_range(idx, step_from, step_to):
            last_llm_step = idx

    items: list[dict[str, Any]] = []
    step = 0
    previous_msgs: list[Message] = []
    prev_main_llm_tokens: int | None = None

    for run in sig[1:]:
        step += 1
        is_compaction = False

        if run.run_type == RunType.LLM:
            if run.parent_run_id not in tool_ids:
                cur_pt = run.prompt_tokens or run.total_tokens or 0
                if prev_main_llm_tokens is not None and cur_pt < prev_main_llm_tokens * (
                    1.0 - _DROP_RATIO
                ):
                    is_compaction = True
                prev_main_llm_tokens = cur_pt

            if mode == "full":
                body, previous_msgs = _narrative_llm_step_full(run, previous_msgs)
            else:
                body, previous_msgs = _narrative_llm_step_compact(run, previous_msgs, step)
            if _step_out_of_range(step, step_from, step_to):
                continue
            item: dict[str, Any] = {
                "type": "llm",
                "tokens": run.total_tokens,
                "header": (
                    f"## Step {step} — llm {run.name} "
                    f"(tokens={run.total_tokens}, latency={_latency_s(run)}s, id={run.id})"
                ),
                "body": body,
            }
        elif run.run_type == RunType.TOOL:
            if _step_out_of_range(step, step_from, step_to):
                continue
            args = _truncate(
                json.dumps(run.inputs, default=str, ensure_ascii=False), _MAX_TOOL_CHARS
            )
            result = _truncate(tool_result_text(run), _MAX_TOOL_CHARS)
            item = {
                "type": "tool",
                "header": f"## Step {step} — tool {run.name} (latency={_latency_s(run)}s, id={run.id})",
                "args": args,
                "result": result,
                "body": f"args: {args}\nresult: {result}",
            }
        else:
            if _step_out_of_range(step, step_from, step_to):
                continue
            output = _truncate(
                json.dumps(run.outputs, default=str, ensure_ascii=False), _MAX_TOOL_CHARS
            )
            item = {
                "type": "node",
                "header": f"## Step {step} — node {run.name} (latency={_latency_s(run)}s, id={run.id})",
                "output": output,
                "body": f"output: {output}",
            }

        if run.error:
            item["body"] = f"{item['body']}\n{_format_error(str(run.error), 1000)}"

        item.update(
            {
                "step": step,
                "run_id": run.id,
                "name": run.name,
                "latency_s": _latency_s(run),
                "status": run.status,
                "error": run.error,
                "is_compaction": is_compaction,
            }
        )
        items.append(item)

    first_step = items[0]["step"] if items else 1
    run_by_id = {r.id: r for r in sig}
    for item in items:
        item["priority"] = _step_priority(
            run_by_id[item["run_id"]],
            item["step"],
            first_step=first_step,
            last_llm_step=last_llm_step,
            tool_ids=tool_ids,
            is_compaction_step=item["is_compaction"],
        )
        item["framing"] = max(len(item["header"]) + 4, _BUDGET_JSON_META_CHARS)
        item["estimated_tokens"] = max(
            (item["framing"] + len(item["body"])) // _CHARS_PER_TOKEN, 1
        )
    return items, step


def _omitted_batches(
    items: list[dict[str, Any]], kept: dict[int, int | None]
) -> list[list[dict[str, Any]]]:
    """Group omitted items into contiguous chronological batches."""
    batches: list[list[dict[str, Any]]] = []
    for item in items:
        if item["step"] in kept:
            continue
        if batches and batches[-1][-1]["step"] == item["step"] - 1:
            batches[-1].append(item)
        else:
            batches.append([item])
    return batches


def _omission_cost(
    batches: list[list[dict[str, Any]]], compact_markers: bool = False
) -> int:
    """Canonical chars charged for omissions: one marker per contiguous batch
    plus a per-step entry covering the JSON omitted_steps entries."""
    return sum(
        len(_format_omitted_marker(b, compact=compact_markers))
        + _BUDGET_OMITTED_STEP_CHARS * len(b)
        for b in batches
    )


def _budget_partition(
    items: list[dict[str, Any]],
    char_budget: int,
    reserved_chars: int,
) -> tuple[dict[int, int | None], bool]:
    """Select steps by (priority, step) into the canonical char budget.

    Returns ({step: payload char limit or None}, compact_markers). Omission
    overhead is charged against the same budget: first the partition tries
    compact markers, then evicts lowest-priority non-anchor kept steps until
    omissions fit. Both renderers call this on the same items, so markdown
    and JSON keep identical step sets.
    """
    target = max(char_budget - reserved_chars, 0)
    kept: dict[int, int | None] = {}
    kept_costs: dict[int, int] = {}
    for item in sorted(items, key=lambda x: (x["priority"], x["step"])):
        full_cost = item["framing"] + len(item["body"])
        if full_cost <= target:
            kept[item["step"]] = None
            kept_costs[item["step"]] = full_cost
            target -= full_cost
        else:
            avail = target - item["framing"]
            if avail >= _MIN_PAYLOAD_CHARS:
                kept[item["step"]] = avail
                kept_costs[item["step"]] = item["framing"] + avail
                target = 0
            break

    batches = _omitted_batches(items, kept)
    compact_markers = _omission_cost(batches) > target
    priorities = {i["step"]: i["priority"] for i in items}
    anchors = {
        items[0]["step"] if items else 1,
        max((i["step"] for i in items if i["type"] == "llm"), default=0),
    }

    while kept and _omission_cost(_omitted_batches(items, kept), compact_markers) > target:
        victim = max(kept, key=lambda s: (priorities[s], s not in anchors, s))
        target += kept_costs.pop(victim)
        del kept[victim]
    return kept, compact_markers


def _payload_for(item: dict[str, Any], limit: int | None, field: str) -> str:
    """Truncate an item's JSON payload field to the shared budget limit.

    The canonical payload is item["body"]; `field` is the JSON-side variable
    text ("body", "result", or "output") with its fixed prefix deducted so
    canonical cost accounting stays identical across renderers.
    """
    text = item[field] if field != "body" else item["body"]
    if limit is None:
        return text
    prefix_len = len(item["body"]) - len(text)
    return _truncate(text, max(limit - prefix_len, 0))


def _prepare_narrative(
    runs: list[Run],
    mode: str,
    step_from: int | None,
    step_to: int | None,
    budget_tokens: int | None,
) -> dict[str, Any] | None:
    """Shared preprocessing for build_narrative / narrative_data.

    Returns None for empty traces; otherwise {root, task, items, total_steps,
    kept} where kept is None without --budget or {step: payload limit}.
    """
    sig = significant_runs(runs)
    if not sig:
        return None

    root = sig[0]
    task = "\n".join(f"[{m.role}] {m.text}" for m in root.input_messages) or json.dumps(
        root.inputs, default=str
    )
    if budget_tokens is not None:
        task = _truncate(task, max((budget_tokens * _CHARS_PER_TOKEN) // 3, 200))

    items, total_steps = _collect_narrative_steps(sig, mode, step_from, step_to)
    kept = None
    compact_markers = False
    if budget_tokens is not None:
        kept, compact_markers = _budget_partition(
            items,
            budget_tokens * _CHARS_PER_TOKEN,
            len(task) + _BUDGET_BASE_OVERHEAD,
        )
    return {
        "root": root,
        "task": task,
        "items": items,
        "total_steps": total_steps,
        "kept": kept,
        "compact_markers": compact_markers,
    }


def build_narrative(
    runs: list[Run],
    mode: str = "compact",
    *,
    step_from: int | None = None,
    step_to: int | None = None,
    budget_tokens: int | None = None,
) -> str:
    """Chronological story of the trace with message deltas per LLM call.

    Args:
        runs: canonical Run objects.
        mode: "compact" (default) — per-index diff, collapses unchanged messages
              even when earlier messages changed (e.g. dynamic system prompts).
              "full" — common-prefix diff, original behavior for humans skimming.
        step_from: if given, only show steps >= step_from (1-based narrative steps).
        step_to: if given, only show steps <= step_to.
        budget_tokens: if given, fill a fixed token budget (chars/4) in priority
              order (anchor/error/subagent/compaction steps -> LLM steps -> tool
              outputs) and emit kept steps in chronological order with explicit
              omission markers charged against the same budget.
    """
    prep = _prepare_narrative(runs, mode, step_from, step_to, budget_tokens)
    if prep is None:
        return "# Narrative\n\n(empty trace)\n"

    root = prep["root"]
    items = prep["items"]
    kept = prep["kept"]
    total_steps = prep["total_steps"]

    sections = [
        f"# Narrative — trace {root.trace_id}",
        "",
        f"status={root.status} total_tokens={root.total_tokens} latency={_latency_s(root)}s",
        "",
        "## Task",
        "",
        prep["task"],
    ]

    if kept is None:
        for item in items:
            sections += ["", item["header"], "", item["body"]]
    else:
        batch: list[dict[str, Any]] = []
        omitted_total = 0
        compact = prep["compact_markers"]
        for item in items:
            if item["step"] in kept:
                if batch:
                    sections += ["", _format_omitted_marker(batch, compact=compact)]
                    batch = []
                limit = kept[item["step"]]
                body = item["body"] if limit is None else _truncate(item["body"], limit)
                sections += ["", item["header"], "", body]
            else:
                batch.append(item)
                omitted_total += 1
        if batch:
            sections += ["", _format_omitted_marker(batch, compact=compact)]

        sections.insert(
            3,
            f"(priority budget: {budget_tokens} tokens, kept "
            f"{len(kept)}/{len(items)} steps, omitted {omitted_total})",
        )

    if step_from is not None or step_to is not None:
        lo = step_from or 1
        hi = step_to or total_steps
        sections.insert(
            3,
            f"(showing steps {lo}-{hi} of {total_steps})",
        )

    return "\n".join(sections) + "\n"


def _step_out_of_range(step: int, step_from: int | None, step_to: int | None) -> bool:
    """True if step should be skipped based on the range filter."""
    if step_from is not None and step < step_from:
        return True
    if step_to is not None and step > step_to:
        return True
    return False


def narrative_data(
    runs: list[Run],
    mode: str = "compact",
    *,
    step_from: int | None = None,
    step_to: int | None = None,
    budget_tokens: int | None = None,
) -> dict[str, Any]:
    """Structured narrative for JSON output (composable contract).

    Returns a dict with trace_id, root task, and a list of steps. Each step
    is a dict with run metadata and either message deltas (LLM), args/result
    (tool), or outputs (chain node).

    Args:
        step_from: if given, only include steps >= step_from (1-based narrative steps).
        step_to: if given, only include steps <= step_to.
        budget_tokens: if given, filter/trim steps using the same priority
            token budget as the markdown view — identical kept step sets.
    """
    prep = _prepare_narrative(runs, mode, step_from, step_to, budget_tokens)
    if prep is None:
        return {"trace_id": runs[0].trace_id if runs else "", "empty": True, "steps": []}

    root = prep["root"]
    items = prep["items"]
    kept = prep["kept"]
    total_steps = prep["total_steps"]

    steps_out: list[dict[str, Any]] = []
    omitted_steps: list[dict[str, Any]] = []
    for item in items:
        if kept is not None and item["step"] not in kept:
            omitted_steps.append(
                {
                    "step": item["step"],
                    "type": item["type"],
                    "name": item["name"],
                    "run_id": item["run_id"],
                    "estimated_tokens": item["estimated_tokens"],
                }
            )
            continue
        limit = kept[item["step"]] if kept is not None else None
        base = {
            "step": item["step"],
            "type": item["type"],
            "run_id": item["run_id"],
            "name": item["name"],
            "latency_s": item["latency_s"],
            "status": item["status"],
            "error": item["error"],
        }
        if item["type"] == "llm":
            base["tokens"] = item["tokens"]
            base["body"] = _payload_for(item, limit, "body")
        elif item["type"] == "tool":
            base["args"] = item["args"]
            base["result"] = _payload_for(item, limit, "result")
        else:
            base["output"] = _payload_for(item, limit, "output")
        steps_out.append(base)

    result: dict[str, Any] = {
        "trace_id": root.trace_id,
        "root_status": root.status,
        "root_total_tokens": root.total_tokens,
        "root_latency_s": _latency_s(root),
        "task": prep["task"],
        "mode": mode,
        "total_steps": total_steps,
        "steps": steps_out,
    }
    if budget_tokens is not None:
        result["budget_tokens"] = budget_tokens
        result["omitted_steps"] = omitted_steps
    if step_from is not None or step_to is not None:
        result["step_range"] = {
            "from": step_from or 1,
            "to": step_to or total_steps,
            "total": total_steps,
        }
    return result
