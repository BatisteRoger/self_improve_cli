"""Outcome policy parsing/derivation and experiment receipt storage (synthetic)."""

from __future__ import annotations

import pytest

from self_improve_cli.domain import (
    Assessment,
    FeedbackScore,
    OutcomeSource,
    OutcomeStatus,
    Trace,
    autofill_assessment,
    outcome_from_policy,
    parse_outcome_key,
)
from self_improve_cli.storage import TraceStore


def test_parse_outcome_key_forms():
    p = parse_outcome_key("score")
    assert (p.key, p.threshold, p.source) == ("score", None, "fetch")
    p = parse_outcome_key("score>=0.5", source="gate")
    assert (p.key, p.threshold, p.source) == ("score", 0.5, "gate")


@pytest.mark.parametrize("raw", ["", ">=0.5", "score>=x", "score<=1", "score>0.5"])
def test_parse_outcome_key_rejects(raw):
    with pytest.raises(ValueError):
        parse_outcome_key(raw)


@pytest.mark.parametrize(
    ("raw", "score", "expected"),
    [
        ("k", 1.0, OutcomeStatus.SUCCESS),
        ("k", 0.0, OutcomeStatus.FAIL),
        ("k", 0.7, OutcomeStatus.UNKNOWN),
        ("k>=0.5", 0.5, OutcomeStatus.SUCCESS),
        ("k>=0.5", 0.49, OutcomeStatus.FAIL),
        ("k>=0.5", None, OutcomeStatus.UNKNOWN),
    ],
)
def test_outcome_from_policy(raw, score, expected):
    fb = [FeedbackScore(key="k", score=score)]
    assert outcome_from_policy(fb, parse_outcome_key(raw))[0] == expected


def test_outcome_from_policy_missing_key_or_policy():
    fb = [FeedbackScore(key="other", score=1.0)]
    assert outcome_from_policy(fb, parse_outcome_key("k"))[0] == OutcomeStatus.UNKNOWN
    assert outcome_from_policy(fb, None)[0] == OutcomeStatus.UNKNOWN


def test_autofill_with_policy_and_human_precedence():
    trace = Trace(trace_id="t1", runs=[])
    fb = [FeedbackScore(key="k", score=0.8)]
    filled = autofill_assessment(None, trace, fb, policy=parse_outcome_key("k>=0.5"))
    assert filled is not None
    assert filled.outcome == OutcomeStatus.SUCCESS
    assert filled.outcome_source == OutcomeSource.EVALUATOR
    human = Assessment(trace_id="t1", task="x", outcome_source=OutcomeSource.HUMAN)
    assert autofill_assessment(human, trace, fb, policy=parse_outcome_key("k>=0.5")) is None


def test_experiment_receipt_roundtrip_and_lookup(tmp_path):
    store = TraceStore(data_root=tmp_path)
    receipt = {"session_id": "s-1", "experiment_name": "exp one", "examples": {}}
    store.save_experiment(receipt)
    assert store.find_experiment("s-1") == receipt
    assert store.find_experiment("exp one") == receipt
    assert store.find_experiment("missing") is None


def test_feedback_roundtrip(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assert store.load_feedback("t1") is None
    store.save_feedback("t1", [FeedbackScore(key="k", score=0.5)])
    assert store.load_feedback("t1") == [FeedbackScore(key="k", score=0.5)]
