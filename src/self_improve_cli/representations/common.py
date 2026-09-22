"""Shared helpers for TER builders — truncation, latency, error formatting,
significant-run filtering.

Pure functions over canonical Run objects. No SDK dependency, no LLM calls:
deterministic and recomputable from raw/sanitized data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

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
