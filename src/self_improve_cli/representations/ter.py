"""TER orchestration — build and persist all derived files for a trace."""

from __future__ import annotations

import logging
from typing import Any

from self_improve_cli.domain import Run
from self_improve_cli.representations.narrative import build_narrative
from self_improve_cli.representations.skeleton import build_skeleton

logger = logging.getLogger(__name__)


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
