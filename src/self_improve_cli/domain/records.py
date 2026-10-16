"""Analyst records — subjective metadata stored alongside a trace.

Records are written by the analyst (human or agent), not computed from trace
data. They are cited judgments: an assessment records what the agent was
asked to do and whether it succeeded; a finding records an evidence-backed
observation. Persisted records are the memory that turns single-trace
diagnostics into cross-trace patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from self_improve_cli.domain.trace import Message, Trace


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
class FeedbackScore:
    """A feedback score recorded on a trace by its provider (evaluator verdict, thumbs).

    Only the key and the numeric score are kept. Free-text comments and
    values are not anonymized, so they are never carried into records.
    """

    key: str
    score: float | None = None


# -- Assessment auto-fill (on fetch) ------------------------------------------

_TASK_MAX_CHARS = 500


def _message_role_and_text(m: Any) -> tuple[str, str]:
    """Read role and plain text from a serialized input message dict."""
    if isinstance(m, dict):
        if "kwargs" in m and isinstance(m.get("id"), list) and m["id"]:
            role = str(m["id"][-1])
            content = (m["kwargs"] or {}).get("content")
        else:
            role = str(m.get("type") or m.get("role") or "")
            content = m.get("content")
    else:
        return "", ""
    if isinstance(content, list):
        content = "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return Message(role=role).role, content if isinstance(content, str) else ""


def initial_task(trace: Trace) -> str:
    """Return the first human message text from the root run's inputs, or "".

    Reads the trace as given — call it on the sanitized trace only.
    Long messages are cut to a bounded preview.
    """
    root = next((r for r in trace.runs if r.parent_run_id is None), None)
    if root is None:
        return ""
    msgs = root.inputs.get("messages") or []
    if len(msgs) == 1 and isinstance(msgs[0], list):
        msgs = msgs[0]
    for m in msgs:
        role, text = _message_role_and_text(m)
        text = text.strip()
        if role == "human" and text:
            if len(text) > _TASK_MAX_CHARS:
                text = f"{text[:_TASK_MAX_CHARS]}…[truncated, {len(text)} chars total]"
            return text
    return ""


def outcome_from_feedback(
    feedback: list[FeedbackScore] | None,
) -> tuple[OutcomeStatus, OutcomeSource, str]:
    """Map recorded feedback to (outcome, source, notes). Never guesses.

    Rule: only binary scores (0 or 1) decide. All scored entries 1 -> success,
    all 0 -> fail, both with source `evaluator`. Mixed binary verdicts or any
    non-binary score leave the outcome unknown; the notes cite what was seen.
    Entries without a numeric score are cited but do not vote.
    """
    if feedback is None:
        return (
            OutcomeStatus.UNKNOWN,
            OutcomeSource.UNKNOWN,
            "Trace feedback unavailable; outcome left unknown.",
        )
    if not feedback:
        return OutcomeStatus.UNKNOWN, OutcomeSource.UNKNOWN, ""
    cited = ", ".join(
        f"{f.key}={f.score:g}" if f.score is not None else f"{f.key} (no score)" for f in feedback
    )
    scores = [f.score for f in feedback if f.score is not None]
    if scores and all(s in (0, 1) for s in scores) and len(set(scores)) == 1:
        outcome = OutcomeStatus.SUCCESS if scores[0] == 1 else OutcomeStatus.FAIL
        return (
            outcome,
            OutcomeSource.EVALUATOR,
            f"Auto-filled from trace feedback: {cited}. Rule: all binary scores 1 -> "
            "success, all 0 -> fail.",
        )
    return (
        OutcomeStatus.UNKNOWN,
        OutcomeSource.UNKNOWN,
        f"Trace feedback seen: {cited}. Outcome left unknown: scores are mixed, "
        "non-binary, or missing.",
    )


def autofill_assessment(
    existing: Assessment | None,
    trace: Trace,
    feedback: list[FeedbackScore] | None,
    policy: Any = None,
) -> Assessment | None:
    """Return the assessment to save after a fetch, or None to leave it as is.

    `policy` (an `OutcomePolicy`) replaces the binary feedback rule with the
    declared key/threshold when given.

    Human precedence: an assessment with source `human` is never touched.
    Otherwise only empty fields are filled — a re-fetch never erases analyst
    work. The outcome and its source are filled together, only while both are
    still unknown. `assessed_at` is left to the caller.
    """
    if existing is not None and existing.outcome_source == OutcomeSource.HUMAN:
        return None
    task = initial_task(trace)
    if policy is not None:
        from self_improve_cli.domain.experiment import outcome_from_policy

        outcome, notes = outcome_from_policy(feedback, policy)
        source = (
            OutcomeSource.EVALUATOR if outcome != OutcomeStatus.UNKNOWN else OutcomeSource.UNKNOWN
        )
    else:
        outcome, source, notes = outcome_from_feedback(feedback)
    base = existing or Assessment(trace_id=trace.trace_id, task="")
    merged = replace(base, task=base.task or task)
    if (
        outcome != OutcomeStatus.UNKNOWN
        and base.outcome == OutcomeStatus.UNKNOWN
        and base.outcome_source == OutcomeSource.UNKNOWN
    ):
        merged.outcome, merged.outcome_source = outcome, source
        merged.notes = f"{base.notes}\n{notes}" if base.notes else notes
    elif not base.notes:
        merged.notes = notes
    if merged == base and existing is not None:
        return None
    if existing is None and not merged.task and not feedback:
        return None
    return merged


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
