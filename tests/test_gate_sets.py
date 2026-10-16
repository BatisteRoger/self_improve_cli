"""Experiment-set gate: classification, flips, noise floor, criteria (synthetic)."""

from __future__ import annotations

import pytest

from self_improve_cli.domain import OutcomePolicy
from self_improve_cli.metrics.efficiency import EfficiencySummary
from self_improve_cli.metrics.gate_sets import (
    RunOutcome,
    Side,
    build_gate_sets,
    classify,
    efficiency_criterion,
    gate_sets_data,
    noise_floor,
)

POLICY = OutcomePolicy("k", 0.5)


def _eff(tokens: int | None = 100, latency: int | None = 1000, tools: int = 2):
    return EfficiencySummary(
        tokens=tokens, llm_calls=1, latency_ms=latency, tool_calls={"t": tools}
    )


def _runs(pattern: str, ex: str, side: str, **eff) -> list[RunOutcome]:
    names = {"P": "pass", "F": "fail", "U": "unknown"}
    return [RunOutcome(f"{side}-{ex}-{i}", names[ch], _eff(**eff)) for i, ch in enumerate(pattern)]


def _side(patterns: dict[str, str], label: str = "b", policy=POLICY, conditions=None) -> Side:
    groups = {ex: _runs(p, ex, label) for ex, p in patterns.items()}
    return Side(label, "experiment", label, policy, groups, conditions or {})


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("PPP", "stable_pass"),
        ("FFF", "stable_fail"),
        ("PFP", "unstable"),
        ("P", "single"),
        ("PUP", "unknown"),
        ("", "missing"),
    ],
)
def test_classify(pattern, expected):
    assert classify(_runs(pattern, "e", "s")) == expected


def test_noise_floor_counts_only_classified():
    floor, unstable, classified = noise_floor(_side({"a": "PFP", "b": "PPP", "c": "P", "d": "PU"}))
    assert (floor, unstable, classified) == (0.5, 1, 2)
    assert noise_floor(_side({"a": "P"}))[0] is None


def _eight(baseline_unstable: int, regressions: int, gains: int = 0):
    base = {f"e{i}": "PPP" for i in range(8)}
    cand = dict(base)
    for i in range(baseline_unstable):
        base[f"e{7 - i}"] = "PFP"
    for i in range(regressions):
        cand[f"e{i}"] = "FFF"
    for i in range(gains):
        base[f"e{3 + i}"] = "FFF"
        cand[f"e{3 + i}"] = "PPP"
    return _side(base, "b"), _side(cand, "c")


def test_pass_without_regression():
    data = gate_sets_data(*_eight(1, 0))
    assert (data["verdict"], data["reason_code"]) == ("pass", None)


def test_within_noise_when_rate_equals_floor():
    data = gate_sets_data(*_eight(1, 1))
    assert (data["verdict"], data["reason_code"]) == ("inconclusive", "within_noise")
    q = data["criteria"]["quality"]
    assert q["regression_rate"] == q["noise_floor"] == 0.125


def test_fail_above_floor_and_gains_do_not_offset():
    data = gate_sets_data(*_eight(1, 2, gains=2))
    assert data["verdict"] == "fail"
    assert data["criteria"]["quality"]["gains"] == 2
    assert data["criteria"]["quality"]["transitions"]["stable_pass"]["stable_fail"] == 2
    rep = data["representative_regressions"]
    assert rep and rep[0]["next"][0].startswith("self-improve diff ")


def test_zero_floor_single_regression_fails():
    assert gate_sets_data(*_eight(0, 1))["verdict"] == "fail"


def test_k1_is_no_noise_estimate_never_pairwise():
    data = gate_sets_data(_side({"a": "P"}, "b"), _side({"a": "F"}, "c"))
    assert (data["verdict"], data["reason_code"]) == ("inconclusive", "no_noise_estimate")


def test_unknown_and_missing_examples():
    data = gate_sets_data(_side({"a": "PUP", "b": "PPP"}, "b"), _side({"a": "PPP"}, "c"))
    assert data["reason_code"] == "quality_unknown"
    reasons = {r["example_id"]: r["reason_code"] for r in data["examples"]}
    assert reasons == {"a": "quality_unknown", "b": "quality_unknown"}


def test_policy_mismatch_is_quality_unknown():
    b = _side({"a": "PPP"}, "b", policy=OutcomePolicy("k", 0.5))
    c = _side({"a": "PPP"}, "c", policy=OutcomePolicy("k", 0.7))
    data = gate_sets_data(b, c)
    assert data["reason_code"] == "quality_unknown"
    policy = next(
        f for f in data["criteria"]["conditions"]["facts"] if f["name"] == "outcome_policy"
    )
    assert policy["state"] == "differs"


def test_efficiency_degraded_is_tradeoff_and_conjunctive():
    base = Side("b", "experiment", "b", POLICY, {"a": _runs("FFF", "a", "b", tokens=100)})
    cand = Side("c", "experiment", "c", POLICY, {"a": _runs("PPP", "a", "c", tokens=500)})
    data = gate_sets_data(base, cand)
    assert data["criteria"]["quality"]["status"] == "pass"
    assert (data["verdict"], data["reason_code"]) == ("inconclusive", "efficiency_tradeoff")


def test_missing_tokens_are_unavailable_not_zero():
    base = Side("b", "experiment", "b", POLICY, {"a": _runs("PP", "a", "b", tokens=None)})
    cand = Side("c", "experiment", "c", POLICY, {"a": _runs("PP", "a", "c", tokens=0)})
    tokens = efficiency_criterion(base, cand)["metrics"]["tokens"]
    assert tokens["unavailable"] == 1 and tokens["median_delta"] is None


def test_strict_conditions_and_unavailable():
    b = _side({"a": "PPP"}, "b", conditions={"git_commit": "aaa", "models": None})
    c = _side({"a": "PPP"}, "c", conditions={"git_commit": "bbb", "models": ["m"]})
    assert gate_sets_data(b, c)["verdict"] == "pass"
    data = gate_sets_data(b, c, strict_conditions=True)
    assert (data["verdict"], data["reason_code"]) == ("inconclusive", "conditions_differ")
    states = {f["name"]: f["state"] for f in data["criteria"]["conditions"]["facts"]}
    assert states["git_commit"] == "differs" and states["models"] == "unavailable"


def test_reason_priority_quality_before_efficiency():
    base = Side("b", "experiment", "b", POLICY, {"a": _runs("P", "a", "b", tokens=1)})
    cand = Side("c", "experiment", "c", POLICY, {"a": _runs("P", "a", "c", tokens=9)})
    assert gate_sets_data(base, cand)["reason_code"] == "no_noise_estimate"


def test_markdown_render():
    out = build_gate_sets(gate_sets_data(*_eight(1, 2)))
    assert out.startswith("# Gate (sets): FAIL")
    assert "| stable_pass | 5 | 0 | 2 |" in out
    assert "Representative regressions:" in out
