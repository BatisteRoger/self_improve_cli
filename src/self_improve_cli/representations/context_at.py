"""L3 context-at — what the model saw at step N (bounded, selective)."""

from __future__ import annotations

import json
from typing import Any

from self_improve_cli.domain import Message, Run, RunType
from self_improve_cli.representations.common import (
    _MAX_TOOL_CHARS,
    _latency_s,
    _message_signature,
    _truncate,
    significant_runs,
)

# Default per-message text preview length for the bounded view.
_CONTEXT_AT_PREVIEW = 400


def _main_loop_llm_runs_for_context(runs: list[Run]) -> list[Run]:
    """Main-loop LLM runs, in chronological order.

    Mirrors context_metrics._main_loop_llm_runs but lives in representations
    so context-at does not depend on the metrics layer. Nested LLM runs
    (children of tool runs) are excluded — they have their own context window.
    """
    sig = significant_runs(runs)
    tool_ids = {r.id for r in sig if r.run_type == RunType.TOOL}
    return [r for r in sig if r.run_type == RunType.LLM and r.parent_run_id not in tool_ids]


def _msg_preview(msg: Message, limit: int = _CONTEXT_AT_PREVIEW) -> str:
    """Bounded preview of a message: role + truncated text + tool calls."""
    text = _truncate(msg.text, limit)
    parts = [f"[{msg.role}] {text}"]
    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = json.dumps(tc.args, default=str, ensure_ascii=False)
            parts.append(f"  -> tool_call {tc.name}({_truncate(args, _MAX_TOOL_CHARS)}) id={tc.id}")
    if msg.tool_call_id:
        parts.append(f"  (tool_call_id={msg.tool_call_id})")
    return "\n".join(parts)


def context_at(
    runs: list[Run],
    step: int,
    *,
    from_step: int | None = None,
    to_step: int | None = None,
    inputs_only: bool = False,
    outputs_only: bool = False,
    tool_call_id: str | None = None,
    full: bool = False,
) -> str:
    """Bounded view of what the model saw at a given main-loop step.

    Args:
        runs: canonical Run objects.
        step: 0-based index into the main-loop LLM runs.
        from_step / to_step: if both given, show a diff of the message list
            between from_step and to_step instead of a single step.
        inputs_only: if True, show only input messages (no output).
        outputs_only: if True, show only the output (no input messages).
        tool_call_id: if given, show only the tool result message matching
            this call id (and the preceding AI tool_call for context).
        full: if True, do not truncate message text.

    Raises ValueError if the step is out of range (recoverable: lists valid steps).
    """
    llm_runs = _main_loop_llm_runs_for_context(runs)
    if not llm_runs:
        return "# Context-at\n\n(no main-loop LLM runs in this trace)\n"

    # Range mode: diff two steps.
    if from_step is not None and to_step is not None:
        return _context_at_diff(runs, llm_runs, from_step, to_step, full=full)

    if step < 0 or step >= len(llm_runs):
        valid = ", ".join(str(i) for i in range(len(llm_runs)))
        raise ValueError(
            f"Step {step} is out of range. Valid steps: 0..{len(llm_runs) - 1} ({valid})."
        )

    run = llm_runs[step]
    lines = [
        f"# Context-at step {step} — {run.name} (id={run.id})",
        "",
        f"status={run.status} tokens={run.total_tokens} "
        f"latency={_latency_s(run)}s error={run.error}",
        "",
    ]

    msgs = run.input_messages
    if tool_call_id:
        msgs = _filter_messages_by_tool_call_id(msgs, tool_call_id)
        lines.append(f"## Filtered to tool_call_id={tool_call_id}")
        lines.append("")
        if not msgs:
            lines.append(f"No message with tool_call_id={tool_call_id} found at step {step}.")
            lines.append(
                "Available tool_call_ids at this step: " + ", ".join(_tool_call_ids_at_step(run))
                or "(none)"
            )
            return "\n".join(lines) + "\n"

    if not outputs_only:
        lines.append("## Input messages")
        lines.append("")
        if not msgs:
            lines.append("(no input messages recorded for this run)")
        else:
            limit = _CONTEXT_AT_PREVIEW if not full else 10**9
            for i, msg in enumerate(msgs):
                lines.append(f"### [{i}] {_msg_preview(msg, limit=limit)}")
                lines.append("")

    if not inputs_only:
        lines.append("## Output")
        lines.append("")
        out = run.output_message
        if out:
            limit = _CONTEXT_AT_PREVIEW if not full else 10**9
            lines.append(_msg_preview(out, limit=limit))
        else:
            lines.append(json.dumps(run.outputs, indent=2, default=str, ensure_ascii=False))

    lines.append("")
    lines.append("## Navigation")
    lines.append("")
    if step > 0:
        prev = llm_runs[step - 1]
        lines.append(
            f"Previous step {step - 1}: {prev.name} id={prev.id}  "
            f"→ `self-improve context-at <trace_id> {step - 1}`"
        )
    if step < len(llm_runs) - 1:
        nxt = llm_runs[step + 1]
        lines.append(
            f"Next step {step + 1}: {nxt.name} id={nxt.id}  "
            f"→ `self-improve context-at <trace_id> {step + 1}`"
        )
    lines.append(f"Full run detail  → `self-improve run-detail <trace_id> {run.id}`")
    return "\n".join(lines) + "\n"


def _tool_call_ids_at_step(run: Run) -> list[str]:
    """All tool_call_ids referenced in a run's input messages."""
    ids: list[str] = []
    for msg in run.input_messages:
        if msg.tool_call_id:
            ids.append(msg.tool_call_id)
        for tc in msg.tool_calls:
            if tc.id:
                ids.append(tc.id)
    return ids


def _filter_messages_by_tool_call_id(msgs: list[Message], tool_call_id: str) -> list[Message]:
    """Keep the AI message that issued the tool_call and the matching tool result."""
    kept: list[Message] = []
    for msg in msgs:
        if any(tc.id == tool_call_id for tc in msg.tool_calls):
            kept.append(msg)
        if msg.tool_call_id == tool_call_id:
            kept.append(msg)
    return kept


def _context_at_diff(
    runs: list[Run],
    llm_runs: list[Run],
    from_step: int,
    to_step: int,
    *,
    full: bool = False,
) -> str:
    """Show what changed in the message list between two steps."""
    for label, s in (("from_step", from_step), ("to_step", to_step)):
        if s < 0 or s >= len(llm_runs):
            raise ValueError(f"{label}={s} is out of range. Valid steps: 0..{len(llm_runs) - 1}.")

    a = llm_runs[from_step]
    b = llm_runs[to_step]
    msgs_a = a.input_messages
    msgs_b = b.input_messages

    lines = [
        f"# Context-at diff: step {from_step} → step {to_step}",
        "",
        f"Step {from_step}: {a.name} id={a.id} ({len(msgs_a)} messages)",
        f"Step {to_step}:   {b.name} id={b.id} ({len(msgs_b)} messages)",
        "",
    ]

    # Per-index comparison (same logic as narrative compact mode).
    max_len = max(len(msgs_a), len(msgs_b))
    added: list[int] = []
    removed: list[int] = []
    changed: list[int] = []
    limit = _CONTEXT_AT_PREVIEW if not full else 10**9

    for i in range(max_len):
        ma = msgs_a[i] if i < len(msgs_a) else None
        mb = msgs_b[i] if i < len(msgs_b) else None
        if ma is None:
            added.append(i)
        elif mb is None:
            removed.append(i)
        elif _message_signature(ma) != _message_signature(mb):
            changed.append(i)

    if not (added or removed or changed):
        lines.append("No differences: the message lists are identical.")
        return "\n".join(lines) + "\n"

    if changed:
        lines.append("## Changed messages")
        lines.append("")
        for i in changed:
            lines.append(f"### [{i}] changed")
            lines.append(f"**Step {from_step}:**")
            lines.append(_msg_preview(msgs_a[i], limit=limit))
            lines.append("")
            lines.append(f"**Step {to_step}:**")
            lines.append(_msg_preview(msgs_b[i], limit=limit))
            lines.append("")

    if added:
        lines.append(f"## Added in step {to_step} (not present at step {from_step})")
        lines.append("")
        for i in added:
            lines.append(f"### [{i}] {_msg_preview(msgs_b[i], limit=limit)}")
            lines.append("")

    if removed:
        lines.append(f"## Present at step {from_step}, gone at step {to_step}")
        lines.append("")
        for i in removed:
            lines.append(f"### [{i}] {_msg_preview(msgs_a[i], limit=limit)}")
            lines.append("")

    # Honest note about inferred vs recorded.
    if len(msgs_b) < len(msgs_a):
        lines.append(
            "Note: the message list shrank. This is consistent with compaction, "
            "but the trace does not record compaction events explicitly — "
            "this is an inferred change, not a recorded fact."
        )
    elif len(msgs_b) > len(msgs_a):
        lines.append("Note: the message list grew. New messages were added between steps.")

    return "\n".join(lines) + "\n"


def context_at_data(
    runs: list[Run],
    step: int,
    *,
    from_step: int | None = None,
    to_step: int | None = None,
    inputs_only: bool = False,
    outputs_only: bool = False,
    tool_call_id: str | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Structured context-at for JSON output (composable contract).

    Returns a dict with step metadata, message list (bounded or full), and
    navigation references. Raises ValueError if the step is out of range.
    """
    llm_runs = _main_loop_llm_runs_for_context(runs)
    if not llm_runs:
        return {"trace_id": runs[0].trace_id if runs else "", "steps": 0, "messages": []}

    if from_step is not None and to_step is not None:
        return _context_at_diff_data(runs, llm_runs, from_step, to_step, full=full)

    if step < 0 or step >= len(llm_runs):
        raise ValueError(f"Step {step} is out of range. Valid steps: 0..{len(llm_runs) - 1}.")

    run = llm_runs[step]
    msgs = run.input_messages
    if tool_call_id:
        msgs = _filter_messages_by_tool_call_id(msgs, tool_call_id)

    limit = _CONTEXT_AT_PREVIEW if not full else 10**9
    messages_out: list[dict[str, Any]] = []
    for i, msg in enumerate(msgs):
        messages_out.append(
            {
                "index": i,
                "role": msg.role,
                "text": _truncate(msg.text, limit) if not full else msg.text,
                "tool_calls": [
                    {"name": tc.name, "args": tc.args, "id": tc.id} for tc in msg.tool_calls
                ],
                "tool_call_id": msg.tool_call_id,
            }
        )

    result: dict[str, Any] = {
        "trace_id": run.trace_id,
        "step": step,
        "total_steps": len(llm_runs),
        "run_id": run.id,
        "run_name": run.name,
        "status": run.status,
        "total_tokens": run.total_tokens,
        "latency_s": _latency_s(run),
        "error": run.error,
        "filtered_to_tool_call_id": tool_call_id,
    }

    if not outputs_only:
        result["messages"] = messages_out

    if not inputs_only:
        out = run.output_message
        if out:
            result["output"] = {
                "role": out.role,
                "text": _truncate(out.text, limit) if not full else out.text,
                "tool_calls": [
                    {"name": tc.name, "args": tc.args, "id": tc.id} for tc in out.tool_calls
                ],
            }
        else:
            result["output"] = run.outputs

    # Navigation references (connectedness).
    nav: dict[str, Any] = {"run_detail_command": f"run-detail <trace_id> {run.id}"}
    if step > 0:
        prev = llm_runs[step - 1]
        nav["previous"] = {"step": step - 1, "run_id": prev.id, "run_name": prev.name}
    if step < len(llm_runs) - 1:
        nxt = llm_runs[step + 1]
        nav["next"] = {"step": step + 1, "run_id": nxt.id, "run_name": nxt.name}
    result["navigation"] = nav
    return result


def _context_at_diff_data(
    runs: list[Run],
    llm_runs: list[Run],
    from_step: int,
    to_step: int,
    *,
    full: bool = False,
) -> dict[str, Any]:
    """Structured diff of message lists between two steps."""
    for label, s in (("from_step", from_step), ("to_step", to_step)):
        if s < 0 or s >= len(llm_runs):
            raise ValueError(f"{label}={s} is out of range. Valid steps: 0..{len(llm_runs) - 1}.")

    a = llm_runs[from_step]
    b = llm_runs[to_step]
    msgs_a = a.input_messages
    msgs_b = b.input_messages
    limit = _CONTEXT_AT_PREVIEW if not full else 10**9

    changed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    max_len = max(len(msgs_a), len(msgs_b))
    for i in range(max_len):
        ma = msgs_a[i] if i < len(msgs_a) else None
        mb = msgs_b[i] if i < len(msgs_b) else None
        if ma is None:
            added.append(_msg_to_dict(mb, i, limit))  # type: ignore[arg-type]
        elif mb is None:
            removed.append(_msg_to_dict(ma, i, limit))
        elif _message_signature(ma) != _message_signature(mb):
            changed.append(
                {
                    "index": i,
                    "from": _msg_to_dict(ma, i, limit),
                    "to": _msg_to_dict(mb, i, limit),
                }
            )

    inferred_note = None
    if len(msgs_b) < len(msgs_a):
        inferred_note = (
            "Message list shrank — consistent with compaction, but the trace "
            "does not record compaction events explicitly. Inferred change."
        )

    return {
        "trace_id": a.trace_id,
        "from_step": from_step,
        "to_step": to_step,
        "from_run_id": a.id,
        "to_run_id": b.id,
        "from_message_count": len(msgs_a),
        "to_message_count": len(msgs_b),
        "changed": changed,
        "added": added,
        "removed": removed,
        "inferred_note": inferred_note,
    }


def _msg_to_dict(msg: Message, index: int, limit: int) -> dict[str, Any]:
    return {
        "index": index,
        "role": msg.role,
        "text": _truncate(msg.text, limit),
        "tool_calls": [{"name": tc.name, "args": tc.args, "id": tc.id} for tc in msg.tool_calls],
        "tool_call_id": msg.tool_call_id,
    }
