"""Tests for persisted findings (finding command + findings.json storage)."""

from __future__ import annotations

import json

import pytest

from self_improve_cli.cli.main import main
from self_improve_cli.domain import Finding, Trace
from self_improve_cli.storage import TraceStore
from tests.helpers import make_root


@pytest.fixture
def saved_trace(tmp_path):
    """Save a minimal sanitized trace under a temp data dir."""
    trace = Trace(
        trace_id="trace-findings-1",
        runs=[make_root(trace_id="trace-findings-1")],
        sanitized=True,
        sanitization_report={"entity_counts": {}, "complete": True, "recognizer_version": "regex"},
        source="test",
    )
    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(trace)
    return trace.trace_id, tmp_path


def _make_finding(finding_id: str = "f1", trace_id: str = "trace-1") -> Finding:
    return Finding(
        id=finding_id,
        trace_id=trace_id,
        title="Agent continued after a 403",
        pattern="tool.ignored_feedback",
        impact="incorrect_result",
        evidence_strength="confirmed",
        triangle_axis="quality",
        evidence=["run-9: 403 response", "run-10: success claimed"],
        assessment="The model received the error and proceeded anyway.",
        fault_locus="model",
        candidate_improvement="harness.fail_on_auth_error",
        created_at="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


def test_save_and_load_findings(tmp_path):
    store = TraceStore(data_root=tmp_path)
    store.save_findings("trace-1", [_make_finding()])

    loaded = store.load_findings("trace-1")
    assert len(loaded) == 1
    f = loaded[0]
    assert f.id == "f1"
    assert f.title == "Agent continued after a 403"
    assert f.pattern == "tool.ignored_feedback"
    assert f.impact == "incorrect_result"
    assert f.evidence_strength == "confirmed"
    assert f.evidence == ["run-9: 403 response", "run-10: success claimed"]
    assert f.fault_locus == "model"
    assert f.candidate_improvement == "harness.fail_on_auth_error"


def test_load_findings_empty_when_missing(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assert store.load_findings("nonexistent") == []


def test_add_finding_appends(tmp_path):
    store = TraceStore(data_root=tmp_path)
    store.add_finding(_make_finding("f1"))
    store.add_finding(_make_finding("f2"))
    loaded = store.load_findings("trace-1")
    assert [f.id for f in loaded] == ["f1", "f2"]


def test_next_finding_id_sequential(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assert store.next_finding_id("trace-1") == "f1"
    store.add_finding(Finding(id="f1", trace_id="trace-1", title="a", pattern="p"))
    store.add_finding(Finding(id="f2", trace_id="trace-1", title="b", pattern="p"))
    assert store.next_finding_id("trace-1") == "f3"


def test_remove_finding(tmp_path):
    store = TraceStore(data_root=tmp_path)
    store.add_finding(_make_finding("f1"))
    store.add_finding(_make_finding("f2"))
    assert store.remove_finding("trace-1", "f1") is True
    assert [f.id for f in store.load_findings("trace-1")] == ["f2"]
    assert store.remove_finding("trace-1", "f9") is False


def test_clear_findings(tmp_path):
    store = TraceStore(data_root=tmp_path)
    store.add_finding(_make_finding("f1"))
    assert store.clear_findings("trace-1") is True
    assert store.load_findings("trace-1") == []
    assert store.clear_findings("trace-1") is False


def test_findings_round_trip_defaults(tmp_path):
    store = TraceStore(data_root=tmp_path)
    store.add_finding(Finding(id="f1", trace_id="trace-1", title="t", pattern="p"))
    f = store.load_findings("trace-1")[0]
    assert f.impact == "no_impact"
    assert f.evidence_strength == "observed"
    assert f.triangle_axis == "quality"
    assert f.evidence == []
    assert f.secondary_patterns == []


# ---------------------------------------------------------------------------
# CLI tests: finding command
# ---------------------------------------------------------------------------


def test_finding_add_and_list(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    rc = main(
        [
            "finding",
            "add",
            trace_id,
            "--title",
            "Agent continued after a 403",
            "--pattern",
            "tool.ignored_feedback",
            "--impact",
            "incorrect_result",
            "--strength",
            "confirmed",
            "--axis",
            "quality",
            "--locus",
            "model",
            "--evidence",
            "run-9: 403 response",
            "--evidence",
            "run-10: success claimed",
            "--assessment",
            "The model received the error.",
            "--candidate",
            "harness.fail_on_auth_error",
            "--data-dir",
            str(data_dir),
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["id"] == "f1"
    assert data["pattern"] == "tool.ignored_feedback"
    assert data["evidence"] == ["run-9: 403 response", "run-10: success claimed"]
    assert data["created_at"] != ""

    rc = main(["finding", "list", trace_id, "--data-dir", str(data_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "f1 | tool.ignored_feedback | incorrect_result | confirmed" in out


def test_finding_list_empty(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    rc = main(["finding", "list", trace_id, "--data-dir", str(data_dir)])
    assert rc == 0
    assert "No findings" in capsys.readouterr().out


def test_finding_list_json(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    main(
        [
            "finding",
            "add",
            trace_id,
            "--title",
            "t",
            "--pattern",
            "p",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()
    rc = main(["finding", "list", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["trace_id"] == trace_id
    assert len(data["findings"]) == 1
    assert data["findings"][0]["id"] == "f1"


def test_finding_add_sequential_ids(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    for _ in range(2):
        main(
            [
                "finding",
                "add",
                trace_id,
                "--title",
                "t",
                "--pattern",
                "p",
                "--data-dir",
                str(data_dir),
            ]
        )
    capsys.readouterr()
    rc = main(["finding", "list", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert [f["id"] for f in data["findings"]] == ["f1", "f2"]


def test_finding_remove(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    main(
        [
            "finding",
            "add",
            trace_id,
            "--title",
            "t",
            "--pattern",
            "p",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()

    rc = main(["finding", "remove", trace_id, "f1", "--data-dir", str(data_dir)])
    assert rc == 0
    rc = main(["finding", "list", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    data = json.loads(capsys.readouterr().out)
    assert data["findings"] == []


def test_finding_remove_invalid_id_recovers(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    rc = main(["finding", "remove", trace_id, "f9", "--data-dir", str(data_dir)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "f9" in err
    assert "finding list" in err


def test_finding_clear(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    main(
        [
            "finding",
            "add",
            trace_id,
            "--title",
            "t",
            "--pattern",
            "p",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()
    rc = main(["finding", "clear", trace_id, "--data-dir", str(data_dir)])
    assert rc == 0
    assert "Cleared" in capsys.readouterr().err


def test_info_includes_findings(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    main(
        [
            "finding",
            "add",
            trace_id,
            "--title",
            "t",
            "--pattern",
            "p",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()
    rc = main(["info", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["findings"] == {"count": 1, "ids": ["f1"]}


def test_info_no_findings_when_absent(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    rc = main(["info", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "findings" not in data
