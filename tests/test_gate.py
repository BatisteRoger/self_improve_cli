"""Tests for the gate command and its verdict rule (offline, no network)."""

import json

import pytest

from self_improve_cli.cli.main import main
from self_improve_cli.domain import Assessment, OutcomeSource, OutcomeStatus, RunType, Trace
from self_improve_cli.metrics.efficiency import efficiency_summary
from self_improve_cli.metrics.gate import (
    EfficiencyStatus,
    QualityStatus,
    Verdict,
    decide_verdict,
    quality_axis,
)
from self_improve_cli.storage import TraceStore
from tests.helpers import make_root, make_run


def _save(store: TraceStore, trace_id: str, tokens: list[int | None], tools: int = 0) -> None:
    runs = [make_root(trace_id=trace_id)]
    runs += [
        make_run(f"l{i}", run_type=RunType.LLM, total_tokens=t, trace_id=trace_id)
        for i, t in enumerate(tokens)
    ]
    runs += [
        make_run(f"t{i}", run_type=RunType.TOOL, name="search", trace_id=trace_id)
        for i in range(tools)
    ]
    store.save_sanitized(
        Trace(
            trace_id=trace_id,
            runs=runs,
            sanitized=True,
            sanitization_report={"entity_counts": {}, "complete": True},
            source="test",
        )
    )


def _assess(store: TraceStore, trace_id: str, outcome: OutcomeStatus) -> None:
    store.save_assessment(
        Assessment(trace_id=trace_id, task="t", outcome=outcome, outcome_source=OutcomeSource.HUMAN)
    )


def _gate(data_dir, fmt: str = "json") -> int:
    return main(["gate", "before", "after", "--data-dir", str(data_dir), "--format", fmt])


@pytest.fixture
def store(tmp_path):
    s = TraceStore(data_root=tmp_path)
    _save(s, "before", [1000, 2000], tools=3)
    return s


# --- verdict rule ------------------------------------------------------------


@pytest.mark.parametrize(
    ("quality", "efficiency", "expected"),
    [
        (QualityStatus.REGRESSED, EfficiencyStatus.IMPROVED, Verdict.FAIL),
        (QualityStatus.UNKNOWN, EfficiencyStatus.IMPROVED, Verdict.INCONCLUSIVE),
        (QualityStatus.IMPROVED, EfficiencyStatus.DEGRADED, Verdict.PASS),
        (QualityStatus.STABLE, EfficiencyStatus.IMPROVED, Verdict.PASS),
        (QualityStatus.STABLE, EfficiencyStatus.EQUAL, Verdict.PASS),
        (QualityStatus.STABLE, EfficiencyStatus.DEGRADED, Verdict.INCONCLUSIVE),
    ],
)
def test_decide_verdict(quality, efficiency, expected):
    assert decide_verdict(quality, efficiency)[0] == expected


def test_quality_axis_ranking_and_unknown():
    def a(o: OutcomeStatus) -> Assessment:
        return Assessment(trace_id="x", task="", outcome=o)

    S, P, F, U = OutcomeStatus  # success, partial, fail, unknown
    assert quality_axis(a(S), a(F)).status == QualityStatus.REGRESSED
    assert quality_axis(a(S), a(P)).status == QualityStatus.REGRESSED
    assert quality_axis(a(F), a(P)).status == QualityStatus.IMPROVED
    assert quality_axis(a(P), a(P)).status == QualityStatus.STABLE
    assert quality_axis(a(S), a(U)).status == QualityStatus.UNKNOWN
    assert quality_axis(None, a(S)).status == QualityStatus.UNKNOWN


# --- CLI ---------------------------------------------------------------------


def test_gate_pass_markdown(store, tmp_path, capsys):
    _save(store, "after", [1000, 1500], tools=2)
    _assess(store, "before", OutcomeStatus.SUCCESS)
    _assess(store, "after", OutcomeStatus.SUCCESS)
    assert _gate(tmp_path, "markdown") == 0
    out = capsys.readouterr().out
    assert "# Gate: PASS" in out
    assert "- Quality: stable" in out
    assert "tokens down: 3000 -> 2500" in out
    assert "self-improve diff before after" in out


def test_gate_fail_on_outcome_regression(store, tmp_path, capsys):
    _save(store, "after", [500], tools=1)
    _assess(store, "before", OutcomeStatus.SUCCESS)
    _assess(store, "after", OutcomeStatus.FAIL)
    assert _gate(tmp_path) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "fail"
    assert data["quality"]["status"] == "regressed"
    assert data["efficiency"]["status"] == "improved"


def test_gate_inconclusive_on_missing_assessment(store, tmp_path, capsys):
    _save(store, "after", [500])
    _assess(store, "before", OutcomeStatus.SUCCESS)
    assert _gate(tmp_path, "markdown") == 2
    out = capsys.readouterr().out
    assert "# Gate: INCONCLUSIVE" in out
    assert "after: no assessment" in out


def test_gate_inconclusive_on_unknown_outcome(store, tmp_path, capsys):
    _save(store, "after", [500])
    _assess(store, "before", OutcomeStatus.UNKNOWN)
    _assess(store, "after", OutcomeStatus.SUCCESS)
    assert _gate(tmp_path) == 2
    assert json.loads(capsys.readouterr().out)["quality"]["status"] == "unknown"


def test_gate_inconclusive_on_cost_regression(store, tmp_path, capsys):
    _save(store, "after", [1000, 2000, 4000], tools=3)
    _assess(store, "before", OutcomeStatus.SUCCESS)
    _assess(store, "after", OutcomeStatus.SUCCESS)
    assert _gate(tmp_path) == 2
    data = json.loads(capsys.readouterr().out)
    assert data["efficiency"]["status"] == "degraded"
    assert data["efficiency"]["metrics"]["tokens"]["direction"] == "up"


@pytest.mark.parametrize("after_tokens", [[None, None], [None, 500]], ids=["all", "partial"])
def test_gate_missing_tokens_are_unavailable_not_improved(store, tmp_path, capsys, after_tokens):
    _save(store, "after", after_tokens, tools=3)
    _assess(store, "before", OutcomeStatus.SUCCESS)
    _assess(store, "after", OutcomeStatus.SUCCESS)
    _gate(tmp_path)
    data = json.loads(capsys.readouterr().out)
    tokens = data["efficiency"]["metrics"]["tokens"]
    assert tokens["direction"] == "unavailable"
    assert tokens["after"] is None
    assert data["efficiency"]["status"] != "improved"
    assert "tokens: unavailable on at least one side, not judged." in data["efficiency"]["reasons"]

    _gate(tmp_path, "markdown")
    out = capsys.readouterr().out
    assert "tokens: unavailable on at least one side, not judged." in out
    assert "tokens down" not in out


def test_efficiency_summary_tokens_none_when_any_llm_run_lacks_usage():
    def llm(idx: str, total: int | None):
        return make_run(idx, run_type=RunType.LLM, total_tokens=total, trace_id="t")

    root = make_root(trace_id="t")
    assert efficiency_summary([root, llm("1", 100), llm("2", 50)]).tokens == 150
    assert efficiency_summary([root, llm("1", 100), llm("2", None)]).tokens is None
    assert efficiency_summary([root, llm("1", None)]).tokens is None


def test_gate_json_shape(store, tmp_path, capsys):
    _save(store, "after", [1000, 2000], tools=3)
    _assess(store, "before", OutcomeStatus.PARTIAL)
    _assess(store, "after", OutcomeStatus.PARTIAL)
    assert _gate(tmp_path) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {
        "trace_before",
        "trace_after",
        "verdict",
        "reason",
        "rule",
        "quality",
        "efficiency",
        "next",
    }
    assert data["verdict"] == "pass"
    assert data["quality"]["before"] == {
        "outcome": "partial",
        "outcome_source": "human",
        "assessed_at": "",
    }
    assert data["efficiency"]["status"] == "equal"
    assert set(data["efficiency"]["metrics"]) == {"tokens", "latency_ms", "tool_calls"}
    assert data["efficiency"]["metrics"]["tool_calls"] == {
        "before": 3,
        "after": 3,
        "delta": 0,
        "pct": 0.0,
        "direction": "equal",
    }


def test_gate_same_trace_is_an_error(store, tmp_path, capsys):
    assert main(["gate", "before", "before", "--data-dir", str(tmp_path)]) == 1
    assert "same" in capsys.readouterr().err


def test_gate_missing_trace_is_an_error(store, tmp_path, capsys):
    assert _gate(tmp_path) == 1
    assert "fetch" in capsys.readouterr().err
