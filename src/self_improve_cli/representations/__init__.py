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
    root_ids = {r.id for r in runs if not r.parent_run_id}

    kept = []
    for run in runs:
        name = run.name
        if any(marker in name for marker in _NOISE_NAME_MARKERS):
            continue
        is_root = not run.parent_run_id
        is_first_level = run.parent_run_id in root_ids
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


def build_narrative(runs: list[Run], mode: str = "compact") -> str:
    """Chronological story of the trace with message deltas per LLM call.

    Args:
        runs: canonical Run objects.
        mode: "compact" (default) — per-index diff, collapses unchanged messages
              even when earlier messages changed (e.g. dynamic system prompts).
              "full" — common-prefix diff, original behavior for humans skimming.
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

    step = 0
    previous_msgs: list[Message] = []
    for run in sig[1:]:
        if run.run_type == RunType.LLM:
            step += 1
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
            outputs = json.dumps(run.outputs, default=str, ensure_ascii=False)
            sections += [
                "",
                f"## Step {step} — node {run.name} (latency={_latency_s(run)}s, id={run.id})",
                "",
                f"output: {_truncate(outputs, _MAX_TOOL_CHARS)}",
            ]
        if run.error:
            sections.append(_format_error(str(run.error), 1000))

    return "\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# L3 — run detail
# ---------------------------------------------------------------------------


def run_detail(runs: list[Run], run_id: str) -> str:
    """Full untruncated context of one run, reconstructed from canonical data."""
    matches = [r for r in runs if str(r.id) == str(run_id)]
    if not matches:
        raise ValueError(f"Run {run_id} not found in trace")
    run = matches[0]

    lines = [
        f"# Run detail — {run.name} ({run.run_type.value}) id={run.id}",
        "",
        f"status={run.status} tokens={run.total_tokens} "
        f"latency={_latency_s(run)}s error={run.error}",
        "",
    ]
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
