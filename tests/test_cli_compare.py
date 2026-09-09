"""Tests for the compare and skill-metrics CLI commands (offline, no network)."""

import json

import pytest

from self_improve_cli.cli.main import main
from self_improve_cli.domain import RunType, Trace
from self_improve_cli.storage import TraceStore
from tests.helpers import make_root, make_run


def _make_llm(idx: str, prompt_tokens: int, trace_id: str) -> object:
    return make_run(
        idx,
        run_type=RunType.LLM,
        prompt_tokens=prompt_tokens,
        trace_id=trace_id,
    )


def _make_tool(idx: str, name: str, args: dict, trace_id: str) -> object:
    return make_run(
        idx,
        run_type=RunType.TOOL,
        name=name,
        trace_id=trace_id,
        inputs=args,
    )


@pytest.fixture
def two_traces(tmp_path):
    """Save two traces: one without skill, one with a skill invocation."""
    # Trace A: no skill (baseline)
    trace_a = Trace(
        trace_id="trace-no-skill",
        runs=[
            make_root(trace_id="trace-no-skill"),
            _make_llm("1", 3500, "trace-no-skill"),
        ],
        sanitized=True,
        sanitization_report={"entity_counts": {}, "complete": True},
        source="test",
    )

    # Trace B: with skill invocation
    trace_b = Trace(
        trace_id="trace-with-skill",
        runs=[
            make_root(trace_id="trace-with-skill"),
            _make_llm("1", 3500, "trace-with-skill"),
            _make_tool(
                "2",
                "read_file",
                {"path": "/priorisation-financiere/SKILL.md"},
                "trace-with-skill",
            ),
            _make_llm("3", 12000, "trace-with-skill"),
        ],
        sanitized=True,
        sanitization_report={"entity_counts": {}, "complete": True},
        source="test",
    )

    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(trace_a)
    store.save_sanitized(trace_b)
    return trace_a.trace_id, trace_b.trace_id, tmp_path


def test_skill_metrics_command_no_skill(tmp_path, capsys):
    """skill-metrics on a trace without skills."""
    trace = Trace(
        trace_id="trace-empty",
        runs=[make_root(trace_id="trace-empty"), _make_llm("1", 100, "trace-empty")],
        sanitized=True,
        sanitization_report={"entity_counts": {}, "complete": True},
        source="test",
    )
    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(trace)
    assert main(["skill-metrics", "trace-empty", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "No skill invocations" in out


def test_skill_metrics_command_with_skill(two_traces, capsys):
    """skill-metrics on a trace with a skill invocation."""
    _, trace_b, data_dir = two_traces
    assert main(["skill-metrics", trace_b, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "priorisation-financiere" in out
    assert "Invocations" in out


def test_compare_command_markdown(two_traces, capsys):
    """compare outputs a markdown table."""
    trace_a, trace_b, data_dir = two_traces
    assert (
        main(["compare", trace_a, trace_b, "--data-dir", str(data_dir), "--format", "markdown"])
        == 0
    )
    out = capsys.readouterr().out
    assert "Trace comparison" in out
    assert "priorisation-financiere" in out
    assert "(none)" in out


def test_compare_command_json(two_traces, capsys):
    """compare outputs structured JSON."""
    trace_a, trace_b, data_dir = two_traces
    assert main(["compare", trace_a, trace_b, "--data-dir", str(data_dir), "--format", "json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["trace_a"] == trace_a
    assert data["trace_b"] == trace_b
    assert data["skills_a"] == []
    assert data["skills_b"] == ["priorisation-financiere"]
    assert data["token_delta"] > 0
    assert data["llm_calls_a"] == 1
    assert data["llm_calls_b"] == 2


def test_compare_command_both_no_skill(tmp_path, capsys):
    """compare two traces without skills."""
    trace_a = Trace(
        trace_id="trace-a",
        runs=[make_root(trace_id="trace-a"), _make_llm("1", 100, "trace-a")],
        sanitized=True,
        sanitization_report={"entity_counts": {}, "complete": True},
        source="test",
    )
    trace_b = Trace(
        trace_id="trace-b",
        runs=[make_root(trace_id="trace-b"), _make_llm("1", 100, "trace-b")],
        sanitized=True,
        sanitization_report={"entity_counts": {}, "complete": True},
        source="test",
    )
    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(trace_a)
    store.save_sanitized(trace_b)
    assert (
        main(["compare", "trace-a", "trace-b", "--data-dir", str(tmp_path), "--format", "json"])
        == 0
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["skills_a"] == []
    assert data["skills_b"] == []
    assert data["token_delta"] == 0


# --- skill-check command ----------------------------------------------------


def test_skill_check_match(two_traces, capsys):
    """skill-check exits 0 when the expected skill was triggered."""
    _, trace_b, data_dir = two_traces
    assert (
        main(
            [
                "skill-check",
                trace_b,
                "--expected",
                "priorisation-financiere",
                "--data-dir",
                str(data_dir),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "priorisation-financiere" in out
    assert "✅" in out


def test_skill_check_mismatch(two_traces, capsys):
    """skill-check exits 1 when the wrong skill was triggered."""
    _, trace_b, data_dir = two_traces
    assert (
        main(
            [
                "skill-check",
                trace_b,
                "--expected",
                "reaction-baisse-marche",
                "--data-dir",
                str(data_dir),
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "❌" in out
    assert "false positive" in out


def test_skill_check_expected_none_match(two_traces, capsys):
    """skill-check exits 0 when no skill was expected and none was triggered."""
    trace_a, _, data_dir = two_traces
    assert main(["skill-check", trace_a, "--expected", "none", "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "none" in out
    assert "✅" in out


def test_skill_check_expected_none_mismatch(two_traces, capsys):
    """skill-check exits 1 when no skill was expected but one was triggered."""
    _, trace_b, data_dir = two_traces
    assert main(["skill-check", trace_b, "--expected", "none", "--data-dir", str(data_dir)]) == 1
    out = capsys.readouterr().out
    assert "❌" in out
    assert "false positive" in out


def test_skill_check_json_match(two_traces, capsys):
    """skill-check --format json returns structured JSON on match."""
    _, trace_b, data_dir = two_traces
    assert (
        main(
            [
                "skill-check",
                trace_b,
                "--expected",
                "priorisation-financiere",
                "--data-dir",
                str(data_dir),
                "--format",
                "json",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["expected"] == "priorisation-financiere"
    assert data["observed"] == ["priorisation-financiere"]
    assert data["match"] is True
    assert data["false_positive"] is None


def test_skill_check_json_mismatch(two_traces, capsys):
    """skill-check --format json returns structured JSON on mismatch."""
    _, trace_b, data_dir = two_traces
    assert (
        main(
            [
                "skill-check",
                trace_b,
                "--expected",
                "reaction-baisse-marche",
                "--data-dir",
                str(data_dir),
                "--format",
                "json",
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["match"] is False
    assert "false_positive" in data
    assert data["false_positive"] is not None
