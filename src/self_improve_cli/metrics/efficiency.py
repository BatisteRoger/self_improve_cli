"""Per-trace efficiency totals — tokens, LLM calls, latency, tool calls.

Shared by `compare` (side-by-side table) and `gate` (before/after verdict).
Totals are computed over significant runs only, like the other metrics.
Latency is wall-clock from the first significant run's start to the latest
end time; it is None when timestamps are missing or unparsable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from self_improve_cli.domain import Run
from self_improve_cli.metrics.tool_metrics import tool_call_counts
from self_improve_cli.representations import significant_runs


@dataclass
class EfficiencySummary:
    """Efficiency totals for one trace."""

    tokens: int
    llm_calls: int
    latency_ms: int | None
    tool_calls: dict[str, int]

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())


def _latency_ms(sig: list[Run]) -> int | None:
    if not sig or not sig[0].start_time:
        return None
    try:
        start = datetime.fromisoformat(sig[0].start_time.replace("Z", "+00:00"))
        end_run = max(sig, key=lambda r: r.end_time or r.start_time or "")
        if not end_run.end_time:
            return None
        end = datetime.fromisoformat(end_run.end_time.replace("Z", "+00:00"))
        return int((end - start).total_seconds() * 1000)
    except (ValueError, TypeError):
        return None


def efficiency_summary(runs: list[Run]) -> EfficiencySummary:
    """Compute efficiency totals for a trace's runs."""
    sig = significant_runs(runs)
    llm = [r for r in sig if r.run_type.value == "llm"]
    return EfficiencySummary(
        tokens=sum(r.total_tokens or 0 for r in llm),
        llm_calls=len(llm),
        latency_ms=_latency_ms(sig),
        tool_calls=tool_call_counts(runs),
    )
