"""Navigation primitives — target-timeline and error-neighborhood views."""

from __future__ import annotations

import json
from typing import Any

from self_improve_cli.domain import Run, RunType
from self_improve_cli.representations.common import (
    _INFRA_ERROR_MARKERS,
    _MAX_TOOL_CHARS,
    _tool_result_text,
    _truncate,
    significant_runs,
)


def _run_matches_target(run: Run, target: str) -> bool:
    """Does a run's inputs/outputs/name mention the target string?"""
    if target in run.name:
        return True
    if target in json.dumps(run.inputs, default=str, ensure_ascii=False):
        return True
    if target in json.dumps(run.outputs, default=str, ensure_ascii=False):
        return True
    return False


def target_timeline(runs: list[Run], target: str, *, compact: bool = False) -> str:
    """Every significant step that touched a target, in chronological order.

    A step "touches" the target if the target string appears in the run's
    name, inputs, or outputs. This is a substring match — the analyst picks
    the target (a file path, a key, a tool name).

    Args:
        compact: if True, show only tool name + status per step, omitting
            full args and results. Useful for overview questions ("did the
            agent keep touching the same target?") on long traces.

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
        lines.append(f"{i + 1}. [{run.run_type.value}] {run.name} id={run.id} — {outcome}")
        if compact:
            lines.append(f"   → `self-improve run-detail <trace_id> {run.id}`")
        else:
            args = json.dumps(run.inputs, default=str, ensure_ascii=False)
            result = (
                _tool_result_text(run)
                if run.run_type == RunType.TOOL
                else json.dumps(run.outputs, default=str, ensure_ascii=False)
            )
            lines.append(f"   args: {_truncate(args, _MAX_TOOL_CHARS)}")
            lines.append(f"   result: {_truncate(result, _MAX_TOOL_CHARS)}")
            lines.append(f"   → `self-improve run-detail <trace_id> {run.id}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def target_timeline_data(runs: list[Run], target: str, *, compact: bool = False) -> dict[str, Any]:
    """Structured target timeline for JSON output (composable contract).

    Args:
        compact: if True, omit args and result from each touch entry.
    """
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": runs[0].trace_id if runs else "", "target": target, "touches": []}

    matches = [r for r in sig if _run_matches_target(r, target)]

    def _touch_entry(i: int, r: Run) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "index": i,
            "run_id": r.id,
            "run_type": r.run_type.value,
            "name": r.name,
            "status": r.status,
            "error": _truncate(str(r.error), 200) if r.error else None,
        }
        if not compact:
            entry["args"] = _truncate(
                json.dumps(r.inputs, default=str, ensure_ascii=False), _MAX_TOOL_CHARS
            )
            entry["result"] = _truncate(
                _tool_result_text(r)
                if r.run_type == RunType.TOOL
                else json.dumps(r.outputs, default=str, ensure_ascii=False),
                _MAX_TOOL_CHARS,
            )
        return entry

    return {
        "trace_id": sig[0].trace_id if sig else "",
        "target": target,
        "touch_count": len(matches),
        "compact": compact,
        "touches": [_touch_entry(i, r) for i, r in enumerate(matches)],
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
