"""L3 run detail — full context of one run, on demand."""

from __future__ import annotations

import json
from typing import Any

from self_improve_cli.domain import Run, RunType
from self_improve_cli.representations.common import (
    _MAX_TOOL_CHARS,
    _format_tool_calls,
    _latency_s,
    _truncate,
    significant_runs,
)


def run_detail(
    runs: list[Run],
    run_id: str,
    *,
    tool_calls_only: bool = False,
    inputs_only: bool = False,
    outputs_only: bool = False,
) -> str:
    """Full untruncated context of one run, reconstructed from canonical data.

    Args:
        tool_calls_only: if True, show only tool calls from the output (LLM runs only).
        inputs_only: if True, show only the input section (no output).
        outputs_only: if True, show only the output section (no input).
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

    # Navigation references (connectedness contract) — shown unless a single
    # section is requested (inputs_only/outputs_only focus on one section).
    if not inputs_only and not outputs_only and (parent or children):
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
        if not outputs_only:
            lines.append("## Input messages")
            for msg in run.input_messages:
                lines += ["", f"### [{msg.role}]", "", msg.text]
                if msg.tool_calls:
                    lines.append(_format_tool_calls(msg.tool_calls))
        if not inputs_only:
            out = run.output_message
            lines += ["", "## Output", ""]
            if out:
                lines.append(out.text)
                if out.tool_calls:
                    lines.append(_format_tool_calls(out.tool_calls))
            else:
                lines.append(json.dumps(run.outputs, indent=2, default=str))
    else:
        if not outputs_only:
            lines += [
                "## Inputs",
                "",
                json.dumps(run.inputs, indent=2, default=str, ensure_ascii=False),
            ]
        if not inputs_only:
            lines += [
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
    runs: list[Run],
    run_id: str,
    *,
    tool_calls_only: bool = False,
    inputs_only: bool = False,
    outputs_only: bool = False,
) -> dict[str, Any]:
    """Structured run detail for JSON output (composable contract).

    Returns a dict with run metadata, related runs (parent/children), and
    either input_messages+output (for LLM runs) or inputs+outputs (for other
    runs). Tool calls are included as structured objects, not prose.

    Args:
        tool_calls_only: if True, return only tool calls from the output (LLM runs).
        inputs_only: if True, omit the output section.
        outputs_only: if True, omit the input section.

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
        if not outputs_only:
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
        if not inputs_only:
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
        if not outputs_only:
            result["inputs"] = run.inputs
        if not inputs_only:
            result["outputs"] = run.outputs

    return result
