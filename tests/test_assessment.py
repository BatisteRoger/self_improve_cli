"""Tests for the task & outcome assessment feature (SLN-17)."""

from __future__ import annotations

import json

import pytest

from self_improve_cli.cli.main import main
from self_improve_cli.domain import Assessment, OutcomeSource, OutcomeStatus, Trace
from self_improve_cli.storage import TraceStore
from tests.helpers import make_root


@pytest.fixture
def saved_trace(tmp_path):
    """Save a minimal sanitized trace under a temp data dir."""
    trace = Trace(
        trace_id="trace-assess-1",
        runs=[make_root(trace_id="trace-assess-1")],
        sanitized=True,
        sanitization_report={"entity_counts": {}, "complete": True, "recognizer_version": "regex"},
        source="test",
    )
    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(trace)
    return trace.trace_id, tmp_path


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


def test_save_and_load_assessment(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assessment = Assessment(
        trace_id="trace-1",
        task="Fix the login bug",
        outcome=OutcomeStatus.SUCCESS,
        outcome_source=OutcomeSource.HUMAN,
        notes="All tests pass.",
        assessed_at="2026-01-01T00:00:00Z",
    )
    store.save_assessment(assessment)

    loaded = store.load_assessment("trace-1")
    assert loaded is not None
    assert loaded.trace_id == "trace-1"
    assert loaded.task == "Fix the login bug"
    assert loaded.outcome == OutcomeStatus.SUCCESS
    assert loaded.outcome_source == OutcomeSource.HUMAN
    assert loaded.notes == "All tests pass."
    assert loaded.assessed_at == "2026-01-01T00:00:00Z"


def test_load_assessment_none_when_missing(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assert store.load_assessment("nonexistent") is None


def test_clear_assessment(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assessment = Assessment(
        trace_id="trace-1",
        task="Test task",
        outcome=OutcomeStatus.UNKNOWN,
    )
    store.save_assessment(assessment)
    assert store.clear_assessment("trace-1") is True
    assert store.load_assessment("trace-1") is None
    # Clearing again returns False (already gone)
    assert store.clear_assessment("trace-1") is False


def test_save_assessment_round_trips_unknown_defaults(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assessment = Assessment(
        trace_id="trace-1",
        task="Test",
    )
    store.save_assessment(assessment)
    loaded = store.load_assessment("trace-1")
    assert loaded is not None
    assert loaded.outcome == OutcomeStatus.UNKNOWN
    assert loaded.outcome_source == OutcomeSource.UNKNOWN
    assert loaded.notes == ""


# ---------------------------------------------------------------------------
# CLI tests: assess command
# ---------------------------------------------------------------------------


def test_assess_set_and_show(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    # Set
    rc = main(
        [
            "assess",
            trace_id,
            "--task",
            "Fix the login bug",
            "--outcome",
            "success",
            "--source",
            "human",
            "--notes",
            "All tests pass.",
            "--data-dir",
            str(data_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["task"] == "Fix the login bug"
    assert data["outcome"] == "success"
    assert data["outcome_source"] == "human"
    assert data["notes"] == "All tests pass."
    assert data["assessed_at"] != ""

    # Show
    rc = main(["assess", trace_id, "--data-dir", str(data_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["task"] == "Fix the login bug"
    assert data["outcome"] == "success"


def test_assess_show_when_none(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    rc = main(["assess", trace_id, "--data-dir", str(data_dir)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No assessment set" in err


def test_assess_clear(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    # Set first
    main(
        [
            "assess",
            trace_id,
            "--task",
            "Test",
            "--outcome",
            "unknown",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()  # clear output

    # Clear
    rc = main(["assess", trace_id, "--clear", "--data-dir", str(data_dir)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "Cleared" in err

    # Verify it's gone
    rc = main(["assess", trace_id, "--data-dir", str(data_dir)])
    assert rc == 1


def test_assess_update_partial(saved_trace, capsys):
    """Updating with only --notes preserves existing task/outcome."""
    trace_id, data_dir = saved_trace
    # Set full assessment
    main(
        [
            "assess",
            trace_id,
            "--task",
            "Original task",
            "--outcome",
            "partial",
            "--source",
            "test",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()

    # Update only notes
    rc = main(
        [
            "assess",
            trace_id,
            "--notes",
            "Updated notes.",
            "--data-dir",
            str(data_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["task"] == "Original task"
    assert data["outcome"] == "partial"
    assert data["notes"] == "Updated notes."


def test_assess_without_task_when_setting_fails(saved_trace, capsys):
    """Setting with --outcome but no --task (and no existing assessment) fails."""
    trace_id, data_dir = saved_trace
    rc = main(
        [
            "assess",
            trace_id,
            "--outcome",
            "success",
            "--data-dir",
            str(data_dir),
        ]
    )
    assert rc == 2  # EXIT_USAGE
    err = capsys.readouterr().err
    assert "--task is required" in err


# ---------------------------------------------------------------------------
# CLI tests: assessment surfacing in skeleton and info
# ---------------------------------------------------------------------------


def test_skeleton_shows_assessment_header(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    main(
        [
            "assess",
            trace_id,
            "--task",
            "Fix the login bug",
            "--outcome",
            "success",
            "--source",
            "human",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()

    rc = main(["skeleton", trace_id, "--data-dir", str(data_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Assessment: success (human)" in out
    assert "Fix the login bug" in out


def test_skeleton_no_assessment_header_when_absent(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    rc = main(["skeleton", trace_id, "--data-dir", str(data_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Assessment:" not in out


def test_skeleton_json_includes_assessment(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    main(
        [
            "assess",
            trace_id,
            "--task",
            "Fix the login bug",
            "--outcome",
            "partial",
            "--source",
            "test",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()

    rc = main(["skeleton", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "assessment" in data
    assert data["assessment"]["task"] == "Fix the login bug"
    assert data["assessment"]["outcome"] == "partial"


def test_skeleton_json_no_assessment_key_when_absent(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    rc = main(["skeleton", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "assessment" not in data


def test_info_includes_assessment(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    main(
        [
            "assess",
            trace_id,
            "--task",
            "Fix the login bug",
            "--outcome",
            "success",
            "--source",
            "human",
            "--data-dir",
            str(data_dir),
        ]
    )
    capsys.readouterr()

    rc = main(["info", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "assessment" in data
    assert data["assessment"]["task"] == "Fix the login bug"
    assert data["assessment"]["outcome"] == "success"


def test_info_no_assessment_when_absent(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    rc = main(["info", trace_id, "--format", "json", "--data-dir", str(data_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "assessment" not in data
