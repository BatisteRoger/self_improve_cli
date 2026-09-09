"""Token-efficient representations (TER) of traces.

Pure functions over canonical Run objects. No SDK dependency, no LLM calls:
deterministic and recomputable from raw/sanitized data.

Granularity levels:
- L1 skeleton:  one line per significant run.
- L2 narrative: chronological story, message deltas per LLM call.
- L3 run detail: full context of one run, on demand.

Adapted from an internal prototype, reworked around the canonical domain model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from self_improve_cli.domain import Message, Run, RunType

logger = logging.getLogger(__name__)

# Heuristics for significant runs (adjust per Target Agent when needed).
_NOISE_NAME_MARKERS = ("Middleware",)
_MAX_MSG_CHARS = 1500
_MAX_TOOL_CHARS = 600

# Error strings that indicate infrastructure/harness interruption, not an
# agent or code defect. When the error text contains one of these markers,
# the skeleton and narrative label it as infra-cancelled so the analyst
# doesn't misdiagnose a manual shutdown or timeout as an agent failure.
_INFRA_ERROR_MARKERS = (
    "CancelledError",
    "asyncio.exceptions.CancelledError",
    "KeyboardInterrupt",
    "TimeoutError",
    "concurrent.futures._base.TimeoutError",
)


def _is_infra_error(error: str) -> bool:
    """Return True if the error string matches a known infra/harness marker."""
    error_lower = error.lower()
    return any(marker.lower() in error_lower for marker in _INFRA_ERROR_MARKERS)


def _format_error(error: str, limit: int) -> str:
    """Format a run error, labeling infra-cancelled errors distinctly."""
    truncated = _truncate(error, limit)
    if _is_infra_error(error):
        return f" ERROR (infra-cancelled, not an agent failure): {truncated}"
    return f" ERROR: {truncated}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latency_s(run: Run) -> float | None:
    start, end = run.start_time, run.end_time
    if not (start and end):
        return None
    try:
        return (
            datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))
        ).total_seconds()
    except ValueError:
        return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[truncated, {len(text)} chars total]"


def _message_signature(msg: Message) -> tuple:
    return (msg.role, msg.text, json.dumps([tc.name for tc in msg.tool_calls], default=str))


def _format_tool_calls(tool_calls: list) -> str:
    lines = []
    for tc in tool_calls:
        args = json.dumps(tc.args, default=str, ensure_ascii=False)
        lines.append(f"-> tool_call {tc.name}({_truncate(args, _MAX_TOOL_CHARS)})")
    return "\n".join(lines)


def _tool_result_text(run: Run) -> str:
    """Best-effort extraction of a tool run's result."""
    outputs = run.outputs or {}
    output = outputs.get("output", outputs)
    if isinstance(output, dict):
        update = output.get("update")
        if isinstance(update, dict):
            for msg in update.get("messages") or []:
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg["content"])
    return json.dumps(output, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Significant-run filter
# ---------------------------------------------------------------------------


def significant_runs(runs: list[Run]) -> list[Run]:
    """Filter raw runs down to the ones that carry meaning.

    Kept: root, llm runs, tool runs, first-level graph nodes.
    Dropped: middleware wrappers and other duplicating chain runs.
    """
    runs = sorted(runs, key=lambda r: r.dotted_order or "")
    root_ids = {str(r.id) for r in runs if not r.parent_run_id}

    kept = []
    for run in runs:
        name = run.name
        if any(marker in name for marker in _NOISE_NAME_MARKERS):
            continue
        is_root = not run.parent_run_id
        is_first_level = str(run.parent_run_id) in root_ids if run.parent_run_id else False
        if is_root or run.run_type in (RunType.LLM, RunType.TOOL) or is_first_level:
            kept.append(run)

    logger.info("Significant runs: kept %d of %d", len(kept), len(runs))
    return kept


# ---------------------------------------------------------------------------
# L1 — skeleton
# ---------------------------------------------------------------------------


def build_skeleton(runs: list[Run]) -> str:
    """One line per significant run: order, type, name, tokens, latency, error."""
    sig = significant_runs(runs)
    if not sig:
        return "# Skeleton\n\n(empty trace)\n"

    root = sig[0]
    lines = [
        f"# Skeleton — trace {root.trace_id}",
        "",
        f"root: {root.name} status={root.status} "
        f"total_tokens={root.total_tokens} latency={_latency_s(root)}s",
        "",
    ]
    for i, run in enumerate(sig):
        depth = len((run.dotted_order or "").split(".")) - 1
        indent = "  " * depth
        error = _format_error(str(run.error), 200) if run.error else ""
        tokens = run.total_tokens
        tokens_part = f" tokens={tokens}" if tokens else ""
        lines.append(
            f"{i:3d}. {indent}[{run.run_type.value}] {run.name}"
            f"{tokens_part} latency={_latency_s(run)}s id={run.id}{error}"
        )
    return "\n".join(lines) + "\n"


def skeleton_data(runs: list[Run]) -> dict[str, Any]:
    """Structured skeleton data for JSON output (composable contract).

    Returns a dict with trace_id, root summary, and a list of significant runs.
    Each run entry includes: index, id, run_type, name, depth, parent_run_id,
    status, total_tokens, latency_s, error.

    Unlike build_skeleton (which returns markdown), this returns structured
    data so an analyst agent can programmatically extract run IDs without
    parsing prose.
    """
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": None, "root": None, "runs": []}

    root = sig[0]
    run_list: list[dict[str, Any]] = []
    for i, run in enumerate(sig):
        depth = len((run.dotted_order or "").split(".")) - 1
        run_list.append(
            {
                "index": i,
                "id": run.id,
                "run_type": run.run_type.value,
                "name": run.name,
                "depth": depth,
                "parent_run_id": run.parent_run_id,
                "status": run.status,
                "total_tokens": run.total_tokens,
                "latency_s": _latency_s(run),
                "error": run.error,
            }
        )

    return {
        "trace_id": root.trace_id,
        "root": {
            "name": root.name,
            "status": root.status,
            "total_tokens": root.total_tokens,
            "latency_s": _latency_s(root),
        },
        "runs": run_list,
    }


# ---------------------------------------------------------------------------
# L2 — narrative
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# L3 — run detail
# ---------------------------------------------------------------------------


def run_detail(runs: list[Run], run_id: str, *, tool_calls_only: bool = False) -> str:
    """Full untruncated context of one run, reconstructed from canonical data.

    Args:
        tool_calls_only: if True, show only tool calls from the output (LLM runs only).
    """
    matches = [r for r in runs if str(r.id) == str(run_id)]
    if not matches:
        # Recoverable error: suggest how to find valid run IDs.
        available = sorted(
            f"  {r.id}  {r.run_type.value}  {r.name}" for r in significant_runs(runs)
        )
        hint = "Run `self-improve skeleton <trace_id>` for a compact overview."
        if available:
            sample = "\n".join(available[:10])
            if len(available) > 10:
                sample += f"\n  ... ({len(available) - 10} more)"
            raise ValueError(
                f"Run {run_id} not found in trace.\n\n"
                f"Available significant runs:\n{sample}\n\n{hint}"
            )
        raise ValueError(f"Run {run_id} not found in trace.\n\n{hint}")
    run = matches[0]

    # Tool-calls-only mode: bounded extraction of tool calls from the output.
    if tool_calls_only:
        return _run_detail_tool_calls_only(run)

    # Find parent and child runs for navigation (connectedness).
    parent = None
    if run.parent_run_id:
        parent = next((r for r in runs if r.id == run.parent_run_id), None)
    children = [r for r in runs if r.parent_run_id == run.id]

    lines = [
        f"# Run detail — {run.name} ({run.run_type.value}) id={run.id}",
        "",
        f"status={run.status} tokens={run.total_tokens} "
        f"latency={_latency_s(run)}s error={run.error}",
        "",
    ]

    # Navigation references (connectedness contract).
    if parent or children:
        lines.append("## Related runs")
        lines.append("")
        if parent:
            lines.append(
                f"Parent: {parent.name} ({parent.run_type.value}) id={parent.id}  "
                f"→ `self-improve run-detail <trace_id> {parent.id}`"
            )
        for child in children:
            lines.append(
                f"Child:  {child.name} ({child.run_type.value}) id={child.id}  "
                f"→ `self-improve run-detail <trace_id> {child.id}`"
            )
        lines.append("")
    if run.run_type == RunType.LLM:
        lines.append("## Input messages")
        for msg in run.input_messages:
            lines += ["", f"### [{msg.role}]", "", msg.text]
            if msg.tool_calls:
                lines.append(_format_tool_calls(msg.tool_calls))
        out = run.output_message
        lines += ["", "## Output", ""]
        if out:
            lines.append(out.text)
            if out.tool_calls:
                lines.append(_format_tool_calls(out.tool_calls))
        else:
            lines.append(json.dumps(run.outputs, indent=2, default=str))
    else:
        lines += [
            "## Inputs",
            "",
            json.dumps(run.inputs, indent=2, default=str, ensure_ascii=False),
            "",
            "## Outputs",
            "",
            json.dumps(run.outputs, indent=2, default=str, ensure_ascii=False),
        ]
    return "\n".join(lines) + "\n"


def _run_detail_tool_calls_only(run: Run) -> str:
    """Bounded view: only tool calls from the run's output."""
    lines = [
        f"# Tool calls — {run.name} ({run.run_type.value}) id={run.id}",
        "",
    ]
    if run.run_type != RunType.LLM:
        lines.append("(tool-calls-only is for LLM runs; this run has no model output.)")
        return "\n".join(lines) + "\n"

    out = run.output_message
    if not out or not out.tool_calls:
        lines.append("(no tool calls in this run's output)")
        return "\n".join(lines) + "\n"

    lines.append(f"## {len(out.tool_calls)} tool call(s)")
    lines.append("")
    for tc in out.tool_calls:
        args = json.dumps(tc.args, default=str, ensure_ascii=False)
        lines.append(f"- **{tc.name}**({_truncate(args, _MAX_TOOL_CHARS)}) id={tc.id}")
    return "\n".join(lines) + "\n"


def run_detail_data(
    runs: list[Run], run_id: str, *, tool_calls_only: bool = False
) -> dict[str, Any]:
    """Structured run detail for JSON output (composable contract).

    Returns a dict with run metadata, related runs (parent/children), and
    either input_messages+output (for LLM runs) or inputs+outputs (for other
    runs). Tool calls are included as structured objects, not prose.

    Args:
        tool_calls_only: if True, return only tool calls from the output (LLM runs).

    Raises ValueError if the run is not found (same recovery as run_detail).
    """
    matches = [r for r in runs if str(r.id) == str(run_id)]
    if not matches:
        available = sorted(
            f"  {r.id}  {r.run_type.value}  {r.name}" for r in significant_runs(runs)
        )
        hint = "Run `self-improve skeleton <trace_id>` for a compact overview."
        if available:
            sample = "\n".join(available[:10])
            if len(available) > 10:
                sample += f"\n  ... ({len(available) - 10} more)"
            raise ValueError(
                f"Run {run_id} not found in trace.\n\n"
                f"Available significant runs:\n{sample}\n\n{hint}"
            )
        raise ValueError(f"Run {run_id} not found in trace.\n\n{hint}")
    run = matches[0]

    # Tool-calls-only mode: bounded extraction.
    if tool_calls_only:
        out = run.output_message if run.run_type == RunType.LLM else None
        tool_calls = (
            [{"name": tc.name, "args": tc.args, "id": tc.id} for tc in out.tool_calls]
            if out and out.tool_calls
            else []
        )
        return {
            "id": run.id,
            "trace_id": run.trace_id,
            "run_type": run.run_type.value,
            "name": run.name,
            "tool_calls": tool_calls,
        }

    parent = None
    if run.parent_run_id:
        parent_run = next((r for r in runs if r.id == run.parent_run_id), None)
        if parent_run:
            parent = {
                "id": parent_run.id,
                "name": parent_run.name,
                "run_type": parent_run.run_type.value,
            }
    children = [
        {"id": r.id, "name": r.name, "run_type": r.run_type.value}
        for r in runs
        if r.parent_run_id == run.id
    ]

    result: dict[str, Any] = {
        "id": run.id,
        "trace_id": run.trace_id,
        "run_type": run.run_type.value,
        "name": run.name,
        "status": run.status,
        "total_tokens": run.total_tokens,
        "latency_s": _latency_s(run),
        "error": run.error,
        "parent": parent,
        "children": children,
    }

    if run.run_type == RunType.LLM:
        result["input_messages"] = [
            {
                "role": msg.role,
                "text": msg.text,
                "tool_calls": [
                    {"name": tc.name, "args": tc.args, "id": tc.id} for tc in msg.tool_calls
                ],
                "tool_call_id": msg.tool_call_id,
            }
            for msg in run.input_messages
        ]
        out = run.output_message
        if out:
            result["output"] = {
                "role": out.role,
                "text": out.text,
                "tool_calls": [
                    {"name": tc.name, "args": tc.args, "id": tc.id} for tc in out.tool_calls
                ],
            }
        else:
            result["output"] = run.outputs
    else:
        result["inputs"] = run.inputs
        result["outputs"] = run.outputs

    return result


# ---------------------------------------------------------------------------
# L3 — context-at: what the model saw at step N (bounded, selective)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Navigation primitives: target-timeline + error-neighborhood (SLN-12)
# ---------------------------------------------------------------------------


def _run_matches_target(run: Run, target: str) -> bool:
    """Does a run's inputs/outputs/name mention the target string?"""
    if target in run.name:
        return True
    if target in json.dumps(run.inputs, default=str, ensure_ascii=False):
        return True
    if target in json.dumps(run.outputs, default=str, ensure_ascii=False):
        return True
    return False


def target_timeline(runs: list[Run], target: str) -> str:
    """Every significant step that touched a target, in chronological order.

    A step "touches" the target if the target string appears in the run's
    name, inputs, or outputs. This is a substring match — the analyst picks
    the target (a file path, a key, a tool name).

    Shows the outcome of each touch (status, error, truncated args/result).
    Includes run IDs for drill-down via run-detail.
    """
    sig = significant_runs(runs)
    if not sig:
        return "# Target timeline\n\n(empty trace)\n"

    matches = [r for r in sig if _run_matches_target(r, target)]
    if not matches:
        return (
            f"# Target timeline — '{target}'\n\n"
            f"No significant run touched '{target}' in this trace.\n"
        )

    lines = [
        f"# Target timeline — '{target}'",
        "",
        f"{len(matches)} step(s) touched this target:",
        "",
    ]
    for i, run in enumerate(matches):
        outcome = run.status or "?"
        if run.error:
            outcome += f" error={_truncate(str(run.error), 200)}"
        args = json.dumps(run.inputs, default=str, ensure_ascii=False)
        result = (
            _tool_result_text(run)
            if run.run_type == RunType.TOOL
            else json.dumps(run.outputs, default=str, ensure_ascii=False)
        )
        lines.append(f"{i + 1}. [{run.run_type.value}] {run.name} id={run.id} — {outcome}")
        lines.append(f"   args: {_truncate(args, _MAX_TOOL_CHARS)}")
        lines.append(f"   result: {_truncate(result, _MAX_TOOL_CHARS)}")
        lines.append(f"   → `self-improve run-detail <trace_id> {run.id}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def target_timeline_data(runs: list[Run], target: str) -> dict[str, Any]:
    """Structured target timeline for JSON output (composable contract)."""
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": runs[0].trace_id if runs else "", "target": target, "touches": []}

    matches = [r for r in sig if _run_matches_target(r, target)]
    return {
        "trace_id": sig[0].trace_id if sig else "",
        "target": target,
        "touch_count": len(matches),
        "touches": [
            {
                "index": i,
                "run_id": r.id,
                "run_type": r.run_type.value,
                "name": r.name,
                "status": r.status,
                "error": _truncate(str(r.error), 200) if r.error else None,
                "args": _truncate(
                    json.dumps(r.inputs, default=str, ensure_ascii=False), _MAX_TOOL_CHARS
                ),
                "result": _truncate(
                    _tool_result_text(r)
                    if r.run_type == RunType.TOOL
                    else json.dumps(r.outputs, default=str, ensure_ascii=False),
                    _MAX_TOOL_CHARS,
                ),
            }
            for i, r in enumerate(matches)
        ],
    }


def _is_infra_cancelled(error: str | None) -> bool:
    """Check if an error is an infra-cancelled (not an agent failure)."""
    if not error:
        return False
    return any(marker in error for marker in _INFRA_ERROR_MARKERS)


def error_neighborhood(runs: list[Run], *, window: int = 1) -> str:
    """Steps around each error, with the agent's reaction.

    For each run with an error, shows the run itself plus ``window`` significant
    runs before and after it. This helps verify whether the agent recovered,
    retried, or ignored the error.

    Args:
        runs: canonical Run objects.
        window: number of significant runs to show before and after each error.
    """
    sig = significant_runs(runs)
    if not sig:
        return "# Error neighborhood\n\n(empty trace)\n"

    error_indices = [i for i, r in enumerate(sig) if r.error and not _is_infra_cancelled(r.error)]
    if not error_indices:
        return "# Error neighborhood\n\nNo agent errors in this trace.\n"

    lines = [
        "# Error neighborhood",
        "",
        f"{len(error_indices)} error(s) found. Showing {window} step(s) before and after each.",
        "",
    ]
    for ei in error_indices:
        start = max(0, ei - window)
        end = min(len(sig), ei + window + 1)
        err_run = sig[ei]
        lines.append(f"## Error at step {ei} — {err_run.name} id={err_run.id}")
        lines.append("")
        lines.append(f"**Error:** {_truncate(str(err_run.error), 500)}")
        lines.append("")
        lines.append("### Neighborhood")
        lines.append("")
        for j in range(start, end):
            r = sig[j]
            marker = " **[ERROR]**" if j == ei else ""
            outcome = r.status or "?"
            lines.append(f"- step {j} [{r.run_type.value}] {r.name} id={r.id} — {outcome}{marker}")
            if r.error and j != ei:
                lines.append(f"  also errored: {_truncate(str(r.error), 200)}")
        lines.append("")
        # Reaction: what did the next significant run do?
        if ei + 1 < len(sig):
            nxt = sig[ei + 1]
            lines.append("### Agent reaction (next significant step)")
            lines.append("")
            lines.append(
                f"Next: [{nxt.run_type.value}] {nxt.name} id={nxt.id} — {nxt.status or '?'}"
            )
            lines.append(f"→ `self-improve run-detail <trace_id> {nxt.id}`")
        else:
            lines.append("### Agent reaction")
            lines.append("")
            lines.append("(no further significant steps — the error was the last action)")
        lines.append("")

    return "\n".join(lines) + "\n"


def error_neighborhood_data(runs: list[Run], *, window: int = 1) -> dict[str, Any]:
    """Structured error neighborhood for JSON output (composable contract)."""
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": runs[0].trace_id if runs else "", "errors": []}

    error_indices = [i for i, r in enumerate(sig) if r.error and not _is_infra_cancelled(r.error)]
    errors_out: list[dict[str, Any]] = []
    for ei in error_indices:
        start = max(0, ei - window)
        end = min(len(sig), ei + window + 1)
        neighborhood: list[dict[str, Any]] = []
        for j in range(start, end):
            r = sig[j]
            neighborhood.append(
                {
                    "step": j,
                    "run_id": r.id,
                    "run_type": r.run_type.value,
                    "name": r.name,
                    "status": r.status,
                    "error": _truncate(str(r.error), 500) if r.error else None,
                    "is_the_error": j == ei,
                }
            )
        reaction = None
        if ei + 1 < len(sig):
            nxt = sig[ei + 1]
            reaction = {
                "step": ei + 1,
                "run_id": nxt.id,
                "run_type": nxt.run_type.value,
                "name": nxt.name,
                "status": nxt.status,
            }
        errors_out.append(
            {
                "error_step": ei,
                "run_id": sig[ei].id,
                "error": _truncate(str(sig[ei].error), 500),
                "neighborhood": neighborhood,
                "agent_reaction": reaction,
            }
        )

    return {
        "trace_id": sig[0].trace_id,
        "error_count": len(error_indices),
        "window": window,
        "errors": errors_out,
    }


# ---------------------------------------------------------------------------
# L1 — tools overview & detail
# ---------------------------------------------------------------------------
_SCOPE_LABELS: dict[frozenset[str], str] = {
    frozenset(): "guardrails (no tools)",
}

# Fallback inference when no explicit label matches. We look for known
# tool-name patterns to guess the scope.
_SCOPE_PATTERNS: list[tuple[frozenset[str], str]] = [
    (frozenset({"search_agent", "financial_agent"}), "supervisor"),
    (frozenset({"search_agent"}), "supervisor (search only)"),
    (frozenset({"rag_tool", "web_search_tool_safe"}), "search_agent"),
]


def _extract_tools(run: Run) -> list[dict[str, Any]]:
    """Extract the tool definitions passed to an LLM run.

    LangSmith stores these in extra.extra.invocation_params.tools as a list of
    {"type": "function", "function": {"name", "description", "parameters"}}.
    Returns an empty list if the run has no tools or the path is missing.
    """
    extra = run.extra.get("extra") or {}
    invocation_params = extra.get("invocation_params") or {}
    tools = invocation_params.get("tools") or []
    return tools if isinstance(tools, list) else []


def _tool_names(tools: list[dict[str, Any]]) -> frozenset[str]:
    """Extract a frozenset of tool names from a tool definition list."""
    names = []
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function") or t
            name = fn.get("name")
            if name:
                names.append(name)
    return frozenset(names)


def _infer_scope_label(tool_set: frozenset[str]) -> str:
    """Infer a short human-readable label for a tool-set signature."""
    if tool_set in _SCOPE_LABELS:
        return _SCOPE_LABELS[tool_set]
    for pattern, label in _SCOPE_PATTERNS:
        if pattern.issubset(tool_set):
            return label
    # Fallback: list the first few tool names
    sample = sorted(tool_set)[:3]
    return ", ".join(sample) + ("…" if len(tool_set) > 3 else "")


def build_tools_overview(runs: list[Run]) -> str:
    """Compact matrix of tools available per LLM run, grouped by scope.

    LLM runs are grouped by their tool-set signature. Each group becomes a
    column in the matrix. Tools are rows. An ``X`` marks presence.
    """
    llm_runs = [r for r in runs if r.run_type == RunType.LLM]
    if not llm_runs:
        return "# Tools\n\n(no LLM runs in this trace)\n"

    # Group runs by tool-set signature
    groups: dict[frozenset[str], list[Run]] = {}
    for run in llm_runs:
        tools = _extract_tools(run)
        sig = _tool_names(tools)
        groups.setdefault(sig, []).append(run)

    # Order groups by first appearance (stable)
    ordered_sigs = list(groups.keys())

    # Collect all unique tool names across groups, preserving first-seen order
    all_tools: list[str] = []
    seen: set[str] = set()
    for sig in ordered_sigs:
        for name in sorted(sig):
            if name not in seen:
                all_tools.append(name)
                seen.add(name)

    # Build header
    root = runs[0] if runs else llm_runs[0]
    lines = [
        f"# Tools — trace {root.trace_id}",
        "",
        f"{len(llm_runs)} LLM runs, {len(ordered_sigs)} scope(s), {len(all_tools)} unique tool(s)",
        "",
    ]

    # Column headers: scope label + run count
    col_labels = []
    for sig in ordered_sigs:
        label = _infer_scope_label(sig)
        count = len(groups[sig])
        col_labels.append(f"{label} ({count} run{'s' if count > 1 else ''})")

    # Compute column width (min 4 for "tool")
    col_width = max(len("tool"), max(len(c) for c in col_labels)) if col_labels else 4

    # Header row
    header = f"{'tool':<{col_width}}  " + "  ".join(f"{c:^{col_width}}" for c in col_labels)
    lines.append(header)
    lines.append("-" * len(header))

    # Matrix rows
    for tool_name in all_tools:
        row = f"{tool_name:<{col_width}}  "
        marks = []
        for sig in ordered_sigs:
            mark = "X" if tool_name in sig else " "
            marks.append(f"{mark:^{col_width}}")
        row += "  ".join(marks)
        lines.append(row)

    # Footer: run IDs per scope
    lines.append("")
    lines.append("## Runs per scope")
    lines.append("")
    for sig in ordered_sigs:
        label = _infer_scope_label(sig)
        run_ids = [r.id for r in groups[sig]]
        lines.append(f"**{label}** ({len(run_ids)} run{'s' if len(run_ids) > 1 else ''}):")
        for rid in run_ids:
            lines.append(f"  - {rid}")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_tools_detail(runs: list[Run], tool_name: str | None = None) -> str:
    """Full docstrings and parameter schemas for tools available in the trace.

    Args:
        runs: canonical Run objects.
        tool_name: if given, show only that tool. If None, show all unique tools.
    """
    llm_runs = [r for r in runs if r.run_type == RunType.LLM]
    if not llm_runs:
        return "# Tools (detail)\n\n(no LLM runs in this trace)\n"

    # Collect unique tool definitions: name -> (definition, scope_labels)
    tools_by_name: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for run in llm_runs:
        tools = _extract_tools(run)
        sig = _tool_names(tools)
        scope_label = _infer_scope_label(sig)
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or t
            name = fn.get("name")
            if not name:
                continue
            if name not in tools_by_name:
                tools_by_name[name] = (fn, [])
            if scope_label not in tools_by_name[name][1]:
                tools_by_name[name][1].append(scope_label)

    if tool_name and tool_name not in tools_by_name:
        available = sorted(tools_by_name.keys())
        return (
            f"# Tools (detail)\n\n"
            f"Tool '{tool_name}' not found in this trace.\n"
            f"Available: {', '.join(available)}\n"
        )

    root = runs[0] if runs else llm_runs[0]
    selected = [tool_name] if tool_name else sorted(tools_by_name.keys())

    lines = [f"# Tools (detail) — trace {root.trace_id}", ""]

    for name in selected:
        fn, scopes = tools_by_name[name]
        desc = fn.get("description", "(no description)")
        params = fn.get("parameters", {})

        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"**Scopes:** {', '.join(scopes)}")
        lines.append("")
        lines.append("### Description")
        lines.append("")
        lines.append(desc)
        lines.append("")

        if params and isinstance(params, dict) and params.get("properties"):
            lines.append("### Parameters")
            lines.append("")
            props = params.get("properties", {})
            required = set(params.get("required", []))
            for pname, pschema in props.items():
                req_marker = " (required)" if pname in required else ""
                ptype = pschema.get("type", "?") if isinstance(pschema, dict) else "?"
                pdesc = pschema.get("description", "") if isinstance(pschema, dict) else ""
                lines.append(f"- **{pname}** (`{ptype}`{req_marker}): {pdesc}")
            lines.append("")
        elif params and isinstance(params, dict):
            lines.append("### Parameters")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(params, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_ter(trace_id: str, runs: list[Run], store: Any) -> dict[str, Any]:
    """Build and write all TER files for a trace.

    Args:
        trace_id: The trace ID.
        runs: Canonical Run objects (sanitized or raw).
        store: A TraceStore instance for persistence.

    Returns a small stats dict (sizes and compression ratio).
    """
    from self_improve_cli.metrics.context_metrics import build_context_metrics
    from self_improve_cli.metrics.tool_metrics import build_tool_metrics

    skeleton = build_skeleton(runs)
    narrative_compact = build_narrative(runs, mode="compact")
    narrative_full = build_narrative(runs, mode="full")
    tool_metrics = build_tool_metrics(runs)
    context_metrics = build_context_metrics(runs)

    store.save_ter_file(trace_id, "skeleton.md", skeleton)
    store.save_ter_file(trace_id, "narrative_compact.md", narrative_compact)
    store.save_ter_file(trace_id, "narrative_full.md", narrative_full)
    store.save_ter_file(trace_id, "tool_metrics.md", tool_metrics)
    store.save_ter_file(trace_id, "context_metrics.md", context_metrics)

    # Compression ratio: sanitized trace size vs compact narrative size.
    sanitized_path = store.traces_dir / trace_id / "sanitized.json"
    raw_chars = sanitized_path.stat().st_size if sanitized_path.exists() else 0
    compression_ratio = round(raw_chars / max(len(narrative_compact), 1), 1) if raw_chars else 0.0

    stats = {
        "trace_id": trace_id,
        "sanitized_chars": raw_chars,
        "skeleton_chars": len(skeleton),
        "narrative_compact_chars": len(narrative_compact),
        "narrative_full_chars": len(narrative_full),
        "narrative_chars": len(narrative_compact),
        "narrative_est_tokens": len(narrative_compact) // 4,
        "compression_ratio": compression_ratio,
        "tool_metrics_chars": len(tool_metrics),
        "context_metrics_chars": len(context_metrics),
    }
    logger.info(
        "TER written for trace %s — narrative_compact=%.1fKB (~%d tokens, %.1fx smaller) "
        "narrative_full=%.1fKB skeleton=%.1fKB tool_metrics=%.1fKB context_metrics=%.1fKB",
        trace_id,
        len(narrative_compact) / 1024,
        stats["narrative_est_tokens"],
        compression_ratio,
        len(narrative_full) / 1024,
        len(skeleton) / 1024,
        len(tool_metrics) / 1024,
        len(context_metrics) / 1024,
    )
    return stats
