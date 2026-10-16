"""Before/after gate — a non-regression verdict receipt for one change.

The gate compares a trace recorded before a change with one recorded after
it, on two axes:

- **Quality** — the analyst assessment outcome on each side, ranked
  success > partial > fail. A missing assessment or an `unknown` outcome on
  either side makes quality unknown: the gate never guesses.
- **Efficiency** — tokens, latency, and tool-call count from the same totals
  `compare` uses. Single-trace measurements, no noise tolerance and no
  statistical significance: any increase counts as degraded.

Verdict rule (`decide_verdict`), in order:

1. quality regressed -> fail
2. quality unknown -> inconclusive
3. quality improved -> pass (a quality gain justifies a cost increase;
   efficiency degradations are still listed in the reasons)
4. quality stable and efficiency improved or equal -> pass
5. quality stable and efficiency degraded -> inconclusive (a cost
   regression may be deliberate; the human gate decides)

`pass` means "no regression observed on this pair", not "the change is
good": a stable `fail -> fail` pair with lower cost passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from self_improve_cli.domain import Assessment, OutcomeStatus
from self_improve_cli.metrics.efficiency import EfficiencySummary

OUTCOME_RANK: dict[OutcomeStatus, int] = {
    OutcomeStatus.SUCCESS: 2,
    OutcomeStatus.PARTIAL: 1,
    OutcomeStatus.FAIL: 0,
}

VERDICT_RULE = (
    "quality regressed -> fail; quality unknown -> inconclusive; "
    "quality improved -> pass; quality stable + efficiency improved/equal -> pass; "
    "quality stable + efficiency degraded -> inconclusive"
)


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class ReasonCode(StrEnum):
    """Machine-readable cause of an inconclusive verdict (shared by v1 and v2)."""

    QUALITY_UNKNOWN = "quality_unknown"
    NO_NOISE_ESTIMATE = "no_noise_estimate"
    WITHIN_NOISE = "within_noise"
    CONDITIONS_DIFFER = "conditions_differ"
    EFFICIENCY_TRADEOFF = "efficiency_tradeoff"


class QualityStatus(StrEnum):
    IMPROVED = "improved"
    STABLE = "stable"
    REGRESSED = "regressed"
    UNKNOWN = "unknown"


class EfficiencyStatus(StrEnum):
    IMPROVED = "improved"
    EQUAL = "equal"
    DEGRADED = "degraded"


@dataclass
class AxisResult:
    status: str
    reasons: list[str] = field(default_factory=list)


def _side(assessment: Assessment | None) -> str:
    if assessment is None:
        return "no assessment"
    return f"{assessment.outcome.value} ({assessment.outcome_source.value})"


def quality_axis(before: Assessment | None, after: Assessment | None) -> AxisResult:
    """Compare assessment outcomes. Missing or unknown on either side -> unknown."""
    if (
        before is None
        or after is None
        or before.outcome == OutcomeStatus.UNKNOWN
        or after.outcome == OutcomeStatus.UNKNOWN
    ):
        detail = f"before: {_side(before)}, after: {_side(after)}"
        return AxisResult(
            QualityStatus.UNKNOWN,
            [f"Outcome cannot be judged ({detail}). Record it with `assess`."],
        )
    rank_b, rank_a = OUTCOME_RANK[before.outcome], OUTCOME_RANK[after.outcome]
    change = f"{before.outcome.value} -> {after.outcome.value}"
    if rank_a < rank_b:
        return AxisResult(QualityStatus.REGRESSED, [f"Outcome regressed: {change}."])
    if rank_a > rank_b:
        return AxisResult(QualityStatus.IMPROVED, [f"Outcome improved: {change}."])
    return AxisResult(QualityStatus.STABLE, [f"Outcome stable: {change}."])


def _metric(before: int | None, after: int | None) -> dict[str, Any]:
    if before is None or after is None:
        return {
            "before": before,
            "after": after,
            "delta": None,
            "pct": None,
            "direction": "unavailable",
        }
    delta = after - before
    direction = "down" if delta < 0 else "up" if delta > 0 else "equal"
    pct = round(delta / before * 100, 1) if before else None
    return {"before": before, "after": after, "delta": delta, "pct": pct, "direction": direction}


def efficiency_metrics(
    before: EfficiencySummary, after: EfficiencySummary
) -> dict[str, dict[str, Any]]:
    """Per-metric before/after/delta. Lower is better for every metric."""
    return {
        "tokens": _metric(before.tokens, after.tokens),
        "latency_ms": _metric(before.latency_ms, after.latency_ms),
        "tool_calls": _metric(before.total_tool_calls, after.total_tool_calls),
    }


def efficiency_axis(metrics: dict[str, dict[str, Any]]) -> AxisResult:
    """Any metric up -> degraded; else any down -> improved; else equal.

    Unavailable metrics (e.g. missing timestamps) are excluded and stated.
    """
    reasons: list[str] = []
    for name, m in metrics.items():
        if m["direction"] == "unavailable":
            reasons.append(f"{name}: unavailable on at least one side, not judged.")
        elif m["direction"] != "equal":
            pct = f", {m['pct']:+g}%" if m["pct"] is not None else ""
            reasons.append(f"{name} {m['direction']}: {m['before']} -> {m['after']}{pct}.")
    directions = {m["direction"] for m in metrics.values()}
    if "up" in directions:
        status = EfficiencyStatus.DEGRADED
    elif "down" in directions:
        status = EfficiencyStatus.IMPROVED
    else:
        status = EfficiencyStatus.EQUAL
    return AxisResult(status, reasons)


def decide_verdict(quality: str, efficiency: str) -> tuple[Verdict, str, ReasonCode | None]:
    """Apply the verdict rule (see module docstring).

    Returns (verdict, reason, reason_code); reason_code is set only when the
    verdict is inconclusive.
    """
    if quality == QualityStatus.REGRESSED:
        return Verdict.FAIL, "Quality regressed.", None
    if quality == QualityStatus.UNKNOWN:
        return (
            Verdict.INCONCLUSIVE,
            "Quality unknown on at least one side; no guess.",
            ReasonCode.QUALITY_UNKNOWN,
        )
    if quality == QualityStatus.IMPROVED:
        return Verdict.PASS, "Quality improved; efficiency changes are listed for review.", None
    if efficiency == EfficiencyStatus.DEGRADED:
        return (
            Verdict.INCONCLUSIVE,
            "Quality stable but efficiency degraded; a cost increase may be deliberate.",
            ReasonCode.EFFICIENCY_TRADEOFF,
        )
    return Verdict.PASS, "Quality stable and efficiency improved or equal.", None


def _assessment_summary(a: Assessment | None) -> dict[str, str] | None:
    if a is None:
        return None
    return {
        "outcome": a.outcome.value,
        "outcome_source": a.outcome_source.value,
        "assessed_at": a.assessed_at,
    }


def gate_data(
    trace_before: str,
    trace_after: str,
    assessment_before: Assessment | None,
    assessment_after: Assessment | None,
    efficiency_before: EfficiencySummary,
    efficiency_after: EfficiencySummary,
) -> dict[str, Any]:
    """Build the structured gate receipt."""
    quality = quality_axis(assessment_before, assessment_after)
    metrics = efficiency_metrics(efficiency_before, efficiency_after)
    efficiency = efficiency_axis(metrics)
    verdict, reason, reason_code = decide_verdict(quality.status, efficiency.status)
    return {
        "mode": "pair",
        "trace_before": trace_before,
        "trace_after": trace_after,
        "verdict": verdict.value,
        "reason_code": reason_code.value if reason_code else None,
        "reason": reason,
        "rule": VERDICT_RULE,
        "quality": {
            "status": quality.status,
            "before": _assessment_summary(assessment_before),
            "after": _assessment_summary(assessment_after),
            "reasons": quality.reasons,
        },
        "efficiency": {
            "status": efficiency.status,
            "approximate": True,
            "metrics": metrics,
            "reasons": efficiency.reasons,
        },
        "next": [
            f"self-improve diff {trace_before} {trace_after}",
            f"self-improve compare {trace_before} {trace_after}",
            f"self-improve assess {trace_before}",
            f"self-improve assess {trace_after}",
        ],
    }


def build_gate(data: dict[str, Any]) -> str:
    """Render the gate receipt as compact Markdown."""
    q, e = data["quality"], data["efficiency"]
    lines = [
        f"# Gate: {data['verdict'].upper()}"
        + (f" ({data['reason_code']})" if data.get("reason_code") else ""),
        "",
        f"Before: {data['trace_before']}  After: {data['trace_after']}",
        f"Verdict reason: {data['reason']}",
        "",
        f"- Quality: {q['status']}",
        *(f"  - {r}" for r in q["reasons"]),
        f"- Efficiency (APPROXIMATE, single traces): {e['status']}",
        *(f"  - {r}" for r in e["reasons"]),
        "",
        f"Rule: {data['rule']}",
        "",
        "Details:",
        *(f"- `{cmd}`" for cmd in data["next"]),
    ]
    return "\n".join(lines)
