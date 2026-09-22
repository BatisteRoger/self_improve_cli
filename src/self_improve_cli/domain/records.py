"""Analyst records — subjective metadata stored alongside a trace.

Records are written by the analyst (human or agent), not computed from trace
data. They are cited judgments: an assessment records what the agent was
asked to do and whether it succeeded; a finding records an evidence-backed
observation. Persisted records are the memory that turns single-trace
diagnostics into cross-trace patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class OutcomeStatus(StrEnum):
    """Outcome of an agent execution, as assessed by the analyst or engineer."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAIL = "fail"
    UNKNOWN = "unknown"


class OutcomeSource(StrEnum):
    """Who or what determined the outcome."""

    HUMAN = "human"
    TEST = "test"
    EVALUATOR = "evaluator"
    UNKNOWN = "unknown"


@dataclass
class Assessment:
    """Manual assessment of what an agent was asked to do and whether it succeeded.

    Stored alongside the trace as metadata. The assessment is cited and
    subjective — it records the analyst's or engineer's judgment, not a
    computed metric. "No verification is visible" is different from "the
    result is incorrect" — keep unknown outcomes explicitly unknown.
    """

    trace_id: str
    task: str
    outcome: OutcomeStatus = OutcomeStatus.UNKNOWN
    outcome_source: OutcomeSource = OutcomeSource.UNKNOWN
    notes: str = ""
    assessed_at: str = ""


@dataclass
class Finding:
    """A persisted evidence-backed observation about a trace.

    Mirrors the findings-report format from the analyze-agent skill. Findings
    accumulate across traces so recurring patterns become visible — a pattern
    seen once is a hypothesis, the same pattern across traces is a mechanism.

    Field values follow the controlled vocabularies in
    skills/analyze-agent/references/mechanisms.md. `pattern`,
    `secondary_patterns`, and `fault_locus` are free strings: the vocabulary
    is living and must not be enforced at the storage boundary. `impact`,
    `evidence_strength`, and `triangle_axis` use fixed value sets validated
    at the CLI boundary.
    """

    id: str
    trace_id: str
    title: str
    pattern: str
    secondary_patterns: list[str] = field(default_factory=list)
    impact: str = "no_impact"
    evidence_strength: str = "observed"
    triangle_axis: str = "quality"
    evidence: list[str] = field(default_factory=list)
    assessment: str = ""
    fault_locus: str = ""
    suggested_next_action: str = ""
    candidate_improvement: str = ""
    validation: str = ""
    created_at: str = ""
