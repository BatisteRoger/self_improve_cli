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
    _tool_result_text,
    _truncate,
    significant_runs,
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


def build_narrative(
    runs: list[Run],
    mode: str = "compact",
    *,
    step_from: int | None = None,
    step_to: int | None = None,
) -> str:
    """Chronological story of the trace with message deltas per LLM call.

    Args:
        runs: canonical Run objects.
        mode: "compact" (default) — per-index diff, collapses unchanged messages
              even when earlier messages changed (e.g. dynamic system prompts).
              "full" — common-prefix diff, original behavior for humans skimming.
        step_from: if given, only show steps >= step_from (1-based narrative steps).
        step_to: if given, only show steps <= step_to.
    """
    sig = significant_runs(runs)
    if not sig:
        return "# Narrative\n\n(empty trace)\n"

    root = sig[0]
    root_msgs = root.input_messages
    task = "\n".join(f"[{m.role}] {m.text}" for m in root_msgs) or json.dumps(
        root.inputs, default=str
    )

    sections = [
        f"# Narrative — trace {root.trace_id}",
        "",
        f"status={root.status} total_tokens={root.total_tokens} latency={_latency_s(root)}s",
        "",
        "## Task",
        "",
        task,
    ]

    total_steps = 0
    step = 0
    previous_msgs: list[Message] = []
    for run in sig[1:]:
        if run.run_type == RunType.LLM:
            step += 1
            total_steps = step
            if _step_out_of_range(step, step_from, step_to):
                if mode == "full":
                    _, previous_msgs = _narrative_llm_step_full(run, previous_msgs)
                else:
                    _, previous_msgs = _narrative_llm_step_compact(run, previous_msgs, step)
                continue
            header = (
                f"## Step {step} — llm {run.name} "
                f"(tokens={run.total_tokens}, latency={_latency_s(run)}s, id={run.id})"
            )
            if mode == "full":
                body, previous_msgs = _narrative_llm_step_full(run, previous_msgs)
            else:
                body, previous_msgs = _narrative_llm_step_compact(run, previous_msgs, step)
            sections += ["", header, "", body]
        elif run.run_type == RunType.TOOL:
            step += 1
            total_steps = step
            if _step_out_of_range(step, step_from, step_to):
                continue
            args = json.dumps(run.inputs, default=str, ensure_ascii=False)
            sections += [
                "",
                f"## Step {step} — tool {run.name} (latency={_latency_s(run)}s, id={run.id})",
                "",
                f"args: {_truncate(args, _MAX_TOOL_CHARS)}",
                f"result: {_truncate(_tool_result_text(run), _MAX_TOOL_CHARS)}",
            ]
        else:
            step += 1
            total_steps = step
            if _step_out_of_range(step, step_from, step_to):
                continue
            outputs = json.dumps(run.outputs, default=str, ensure_ascii=False)
            sections += [
                "",
                f"## Step {step} — node {run.name} (latency={_latency_s(run)}s, id={run.id})",
                "",
                f"output: {_truncate(outputs, _MAX_TOOL_CHARS)}",
            ]
        if run.error:
            sections.append(_format_error(str(run.error), 1000))

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
) -> dict[str, Any]:
    """Structured narrative for JSON output (composable contract).

    Returns a dict with trace_id, root task, and a list of steps. Each step
    is a dict with run metadata and either message deltas (LLM), args/result
    (tool), or outputs (chain node).

    Args:
        step_from: if given, only include steps >= step_from (1-based narrative steps).
        step_to: if given, only include steps <= step_to.
    """
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": runs[0].trace_id if runs else "", "empty": True, "steps": []}

    root = sig[0]
    root_msgs = root.input_messages
    task = "\n".join(f"[{m.role}] {m.text}" for m in root_msgs) or json.dumps(
        root.inputs, default=str
    )

    steps_out: list[dict[str, Any]] = []
    total_steps = 0
    step = 0
    previous_msgs: list[Message] = []
    for run in sig[1:]:
        if run.run_type == RunType.LLM:
            step += 1
            total_steps = step
            if mode == "full":
                body, previous_msgs = _narrative_llm_step_full(run, previous_msgs)
            else:
                body, previous_msgs = _narrative_llm_step_compact(run, previous_msgs, step)
            if _step_out_of_range(step, step_from, step_to):
                continue
            steps_out.append(
                {
                    "step": step,
                    "type": "llm",
                    "run_id": run.id,
                    "name": run.name,
                    "tokens": run.total_tokens,
                    "latency_s": _latency_s(run),
                    "status": run.status,
                    "error": run.error,
                    "body": body,
                }
            )
        elif run.run_type == RunType.TOOL:
            step += 1
            total_steps = step
            if _step_out_of_range(step, step_from, step_to):
                continue
            args = json.dumps(run.inputs, default=str, ensure_ascii=False)
            steps_out.append(
                {
                    "step": step,
                    "type": "tool",
                    "run_id": run.id,
                    "name": run.name,
                    "latency_s": _latency_s(run),
                    "status": run.status,
                    "error": run.error,
                    "args": _truncate(args, _MAX_TOOL_CHARS),
                    "result": _truncate(_tool_result_text(run), _MAX_TOOL_CHARS),
                }
            )
        else:
            step += 1
            total_steps = step
            if _step_out_of_range(step, step_from, step_to):
                continue
            outputs = json.dumps(run.outputs, default=str, ensure_ascii=False)
            steps_out.append(
                {
                    "step": step,
                    "type": "node",
                    "run_id": run.id,
                    "name": run.name,
                    "latency_s": _latency_s(run),
                    "status": run.status,
                    "error": run.error,
                    "output": _truncate(outputs, _MAX_TOOL_CHARS),
                }
            )

    result = {
        "trace_id": root.trace_id,
        "root_status": root.status,
        "root_total_tokens": root.total_tokens,
        "root_latency_s": _latency_s(root),
        "task": task,
        "mode": mode,
        "total_steps": total_steps,
        "steps": steps_out,
    }
    if step_from is not None or step_to is not None:
        result["step_range"] = {
            "from": step_from or 1,
            "to": step_to or total_steps,
            "total": total_steps,
        }
    return result
