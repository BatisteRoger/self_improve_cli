"""Experiment sets — outcome policy and the `experiment.json` receipt.

An experiment is a set of runs over a dataset, possibly repeated per
example. The receipt groups stored trace IDs by dataset example and records
which feedback key decides pass/fail (the outcome policy). The policy is
declared by the user, never inferred: a run without the declared key keeps
an `unknown` outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from self_improve_cli.domain.records import FeedbackScore, OutcomeStatus

EXPERIMENT_SCHEMA_VERSION = 1


@dataclass
class OutcomePolicy:
    """Which feedback key decides pass/fail, and an optional `>=` threshold.

    `source` records who declared it: `fetch`, `gate`, or `assessment`
    (outcomes taken from stored assessments, no feedback key).
    """

    key: str | None
    threshold: float | None = None
    source: str = "fetch"

    def describe(self) -> str:
        if self.key is None:
            return f"({self.source})"
        return f"{self.key}>={self.threshold:g}" if self.threshold is not None else self.key

    def same_rule(self, other: OutcomePolicy) -> bool:
        return (self.key, self.threshold) == (other.key, other.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "threshold": self.threshold, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OutcomePolicy | None:
        if not data:
            return None
        return cls(data.get("key"), data.get("threshold"), data.get("source", "fetch"))


def parse_outcome_key(raw: str, source: str = "fetch") -> OutcomePolicy:
    """Parse `KEY` or `KEY>=T`. Any other form raises ValueError."""
    usage = "Use --outcome-key KEY (binary 0/1 score) or --outcome-key 'KEY>=T'."
    key, sep, threshold = raw.partition(">=")
    key = key.strip()
    if not key or any(op in key for op in ("<", ">", "=")):
        raise ValueError(f"Invalid --outcome-key {raw!r}. {usage}")
    if not sep:
        return OutcomePolicy(key, None, source)
    try:
        return OutcomePolicy(key, float(threshold), source)
    except ValueError:
        raise ValueError(f"Invalid threshold in --outcome-key {raw!r}. {usage}") from None


def outcome_from_policy(
    feedback: list[FeedbackScore] | None, policy: OutcomePolicy | None
) -> tuple[OutcomeStatus, str]:
    """Map feedback to (success | fail | unknown, note) under a declared policy.

    Without threshold the score must be exactly 0 or 1. With threshold,
    `score >= T` is success. Missing policy, key, or score -> unknown.
    """
    if policy is None or policy.key is None:
        return OutcomeStatus.UNKNOWN, "No outcome policy declared; outcome left unknown."
    entry = next((f for f in feedback or [] if f.key == policy.key), None)
    score = entry.score if entry else None
    if entry is None or score is None:
        return (
            OutcomeStatus.UNKNOWN,
            f"Feedback key '{policy.key}' not recorded with a numeric score; outcome unknown.",
        )
    rule = f"policy {policy.describe()}, {policy.key}={score:g}"
    if (entry.n or 0) > 1:
        rule += f" (mean of {entry.n} points)"
    if policy.threshold is not None:
        status = OutcomeStatus.SUCCESS if score >= policy.threshold else OutcomeStatus.FAIL
        return status, f"Outcome from {rule}."
    if score in (0, 1):
        status = OutcomeStatus.SUCCESS if score == 1 else OutcomeStatus.FAIL
        return status, f"Outcome from {rule}."
    return (
        OutcomeStatus.UNKNOWN,
        f"Non-binary score ({rule}); declare a threshold with 'KEY>=T'. Outcome unknown.",
    )


@dataclass
class ExperimentRunRef:
    """One experiment run as listed by the source (no trace content)."""

    run_id: str
    trace_id: str
    example_id: str
    status: str | None = None
    feedback: list[FeedbackScore] = field(default_factory=list)


@dataclass
class ExperimentListing:
    """Canonical result of listing an experiment session at the source."""

    name: str
    session_id: str
    dataset_id: str
    conditions: dict[str, Any]
    runs: list[ExperimentRunRef]
    complete: bool = True
