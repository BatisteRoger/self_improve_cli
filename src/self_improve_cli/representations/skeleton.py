"""L1 skeleton — one line per significant run."""

from __future__ import annotations

from typing import Any

from self_improve_cli.domain import Run
from self_improve_cli.representations.common import (
    _format_error,
    _latency_s,
    significant_runs,
)


def build_skeleton(runs: list[Run], *, errors_only: bool = False) -> str:
    """One line per significant run: order, type, name, tokens, latency, error.

    Args:
        errors_only: if True, show only runs with error or cancelled status,
            plus a header noting how many were filtered out. The root summary
            line is always shown (it carries trace-level status).
    """
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

    if errors_only:
        error_runs = [
            (i, run)
            for i, run in enumerate(sig)
            if run.error or (run.status and run.status != "success")
        ]
        if not error_runs:
            lines.append("No errors or cancellations in this trace.")
            return "\n".join(lines) + "\n"
        lines.append(f"{len(error_runs)} error/cancelled run(s) (of {len(sig)} significant runs):")
        lines.append("")
        for i, run in error_runs:
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


def skeleton_data(runs: list[Run], *, errors_only: bool = False) -> dict[str, Any]:
    """Structured skeleton data for JSON output (composable contract).

    Returns a dict with trace_id, root summary, and a list of significant runs.
    Each run entry includes: index, id, run_type, name, depth, parent_run_id,
    status, total_tokens, latency_s, error.

    Unlike build_skeleton (which returns markdown), this returns structured
    data so an analyst agent can programmatically extract run IDs without
    parsing prose.

    Args:
        errors_only: if True, the runs list contains only error/cancelled runs.
            The dict includes `errors_only: true` and `filtered_out` count.
    """
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": None, "root": None, "runs": []}

    root = sig[0]

    def _run_entry(i: int, run: Run) -> dict[str, Any]:
        depth = len((run.dotted_order or "").split(".")) - 1
        return {
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

    if errors_only:
        error_runs = [
            (i, run)
            for i, run in enumerate(sig)
            if run.error or (run.status and run.status != "success")
        ]
        return {
            "trace_id": root.trace_id,
            "root": {
                "name": root.name,
                "status": root.status,
                "total_tokens": root.total_tokens,
                "latency_s": _latency_s(root),
            },
            "errors_only": True,
            "filtered_out": len(sig) - len(error_runs),
            "runs": [_run_entry(i, run) for i, run in error_runs],
        }

    run_list = [_run_entry(i, run) for i, run in enumerate(sig)]
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
