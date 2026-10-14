"""Tests for assessment auto-fill on fetch (offline, no network)."""

import json
from types import SimpleNamespace

import pytest

from self_improve_cli.cli.main import main
from self_improve_cli.domain import (
    Assessment,
    FeedbackScore,
    OutcomeSource,
    OutcomeStatus,
    Trace,
    autofill_assessment,
)
from self_improve_cli.domain.records import initial_task, outcome_from_feedback
from self_improve_cli.storage import TraceStore
from tests.helpers import make_root


def _trace(inputs: dict | None = None, trace_id: str = "t1") -> Trace:
    root = make_root(trace_id=trace_id)
    root.inputs = inputs or {}
    return Trace(trace_id=trace_id, runs=[root], sanitized=True, source="test")


_HUMAN_INPUTS = {
    "messages": [
        {"type": "system", "content": "You are an agent."},
        {"type": "human", "content": "Summarize the report."},
    ]
}


# -- initial_task ---------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        {"messages": [{"role": "user", "content": "Summarize the report."}]},
        {"messages": [[{"type": "human", "content": "Summarize the report."}]]},
        {
            "messages": [
                {
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {"content": [{"type": "text", "text": "Summarize the report."}]},
                }
            ]
        },
        _HUMAN_INPUTS,
    ],
)
def test_initial_task_reads_first_human_message(inputs):
    assert initial_task(_trace(inputs)) == "Summarize the report."


def test_initial_task_empty_when_no_human_message():
    assert initial_task(_trace({"messages": [{"type": "ai", "content": "hi"}]})) == ""
    assert initial_task(_trace({"input": "not a message list"})) == ""
    assert initial_task(Trace(trace_id="t1", runs=[])) == ""


def test_initial_task_is_bounded():
    task = initial_task(_trace({"messages": [{"type": "human", "content": "x" * 2000}]}))
    assert task.endswith("…[truncated, 2000 chars total]")
    assert len(task) < 600


# -- outcome_from_feedback ------------------------------------------------------


def test_feedback_all_ones_is_success():
    outcome, source, notes = outcome_from_feedback(
        [FeedbackScore("correctness", 1.0), FeedbackScore("thumbs", 1.0)]
    )
    assert (outcome, source) == (OutcomeStatus.SUCCESS, OutcomeSource.EVALUATOR)
    assert "correctness=1" in notes and "thumbs=1" in notes


def test_feedback_all_zeros_is_fail():
    outcome, source, notes = outcome_from_feedback([FeedbackScore("correctness", 0.0)])
    assert (outcome, source) == (OutcomeStatus.FAIL, OutcomeSource.EVALUATOR)
    assert "correctness=0" in notes


@pytest.mark.parametrize(
    "feedback",
    [
        [FeedbackScore("correctness", 1.0), FeedbackScore("helpfulness", 0.0)],
        [FeedbackScore("helpfulness", 0.7)],
        [FeedbackScore("label", None)],
    ],
)
def test_ambiguous_feedback_stays_unknown_and_cites(feedback):
    outcome, source, notes = outcome_from_feedback(feedback)
    assert (outcome, source) == (OutcomeStatus.UNKNOWN, OutcomeSource.UNKNOWN)
    assert feedback[0].key in notes
    assert "left unknown" in notes


def test_no_feedback_leaves_notes_empty():
    assert outcome_from_feedback([]) == (OutcomeStatus.UNKNOWN, OutcomeSource.UNKNOWN, "")


def test_unavailable_feedback_is_stated():
    outcome, _, notes = outcome_from_feedback(None)
    assert outcome == OutcomeStatus.UNKNOWN
    assert "unavailable" in notes


# -- autofill_assessment (merge rules) ------------------------------------------


def test_autofill_creates_from_task():
    a = autofill_assessment(None, _trace(_HUMAN_INPUTS), [])
    assert a is not None
    assert a.task == "Summarize the report."
    assert (a.outcome, a.outcome_source, a.notes) == (
        OutcomeStatus.UNKNOWN,
        OutcomeSource.UNKNOWN,
        "",
    )


def test_autofill_nothing_to_fill_returns_none():
    assert autofill_assessment(None, _trace(), []) is None
    assert autofill_assessment(None, _trace(), None) is None


def test_autofill_never_touches_human():
    human = Assessment(
        trace_id="t1", task="", outcome_source=OutcomeSource.HUMAN, outcome=OutcomeStatus.FAIL
    )
    assert autofill_assessment(human, _trace(_HUMAN_INPUTS), [FeedbackScore("c", 1.0)]) is None


def test_autofill_fills_only_empty_fields():
    existing = Assessment(trace_id="t1", task="Analyst task", notes="analyst note")
    a = autofill_assessment(existing, _trace(_HUMAN_INPUTS), [FeedbackScore("c", 1.0)])
    assert a is not None
    assert a.task == "Analyst task"
    assert a.outcome == OutcomeStatus.SUCCESS
    assert a.notes.startswith("analyst note\n")
    assert "c=1" in a.notes


def test_autofill_keeps_existing_outcome():
    existing = Assessment(
        trace_id="t1",
        task="Analyst task",
        outcome=OutcomeStatus.PARTIAL,
        outcome_source=OutcomeSource.TEST,
        notes="from CI",
    )
    assert autofill_assessment(existing, _trace(_HUMAN_INPUTS), [FeedbackScore("c", 0.0)]) is None


# -- fetch integration ----------------------------------------------------------


class _Source:
    feedback: list[FeedbackScore] | None = []
    inputs: dict = _HUMAN_INPUTS

    def __init__(self, project_name=None, **_kwargs):
        self.project_name = project_name

    def fetch_trace(self, trace_id: str, project_id: str | None = None) -> Trace:
        root = make_root(trace_id=trace_id)
        root.inputs = json.loads(json.dumps(self.inputs))
        return Trace(trace_id=trace_id, runs=[root], sanitized=False, source="test")

    def fetch_feedback(self, trace_id: str):
        return self.feedback


def _fetch(tmp_path, monkeypatch, source_cls, capsys) -> dict:
    import self_improve_cli.sources.langsmith as ls_module

    monkeypatch.setattr(ls_module, "LangSmithSource", source_cls)
    rc = main(
        [
            "fetch",
            "trace-x",
            "--anonymizer",
            "regex",
            "--data-dir",
            str(tmp_path),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    return json.loads(capsys.readouterr().out)


def test_fetch_prefills_task(tmp_path, monkeypatch, capsys):
    out = _fetch(tmp_path, monkeypatch, _Source, capsys)
    assert out["assessment"] == "created"
    a = TraceStore(data_root=tmp_path).load_assessment("trace-x")
    assert a is not None
    assert a.task == "Summarize the report."
    assert (a.outcome, a.outcome_source) == (OutcomeStatus.UNKNOWN, OutcomeSource.UNKNOWN)
    assert a.assessed_at


def test_fetch_task_comes_from_sanitized_trace(tmp_path, monkeypatch, capsys):
    class _PiiSource(_Source):
        inputs = {"messages": [{"type": "human", "content": "Email jane.doe@example.com now"}]}

    _fetch(tmp_path, monkeypatch, _PiiSource, capsys)
    a = TraceStore(data_root=tmp_path).load_assessment("trace-x")
    assert a is not None
    assert "jane.doe@example.com" not in a.task
    assert a.task.startswith("Email ")


def test_fetch_with_feedback_sets_evaluator_outcome(tmp_path, monkeypatch, capsys):
    class _FbSource(_Source):
        feedback = [FeedbackScore("correctness", 1.0)]

    _fetch(tmp_path, monkeypatch, _FbSource, capsys)
    a = TraceStore(data_root=tmp_path).load_assessment("trace-x")
    assert a is not None
    assert (a.outcome, a.outcome_source) == (OutcomeStatus.SUCCESS, OutcomeSource.EVALUATOR)
    assert "correctness=1" in a.notes


def test_refetch_keeps_human_assessment(tmp_path, monkeypatch, capsys):
    class _FbSource(_Source):
        feedback = [FeedbackScore("correctness", 0.0)]

    store = TraceStore(data_root=tmp_path)
    human = Assessment(
        trace_id="trace-x",
        task="Human task",
        outcome=OutcomeStatus.SUCCESS,
        outcome_source=OutcomeSource.HUMAN,
        notes="Verified by hand.",
        assessed_at="2026-01-01T00:00:00+00:00",
    )
    path = store.save_assessment(human)
    before = path.read_bytes()

    out = _fetch(tmp_path, monkeypatch, _FbSource, capsys)
    assert out["assessment"] == "kept (source=human)"
    assert path.read_bytes() == before


def test_fetch_feedback_failure_does_not_fail_fetch(tmp_path, monkeypatch, capsys):
    class _BrokenSource(_Source):
        def fetch_feedback(self, trace_id: str):
            raise ConnectionError("down")

    import self_improve_cli.sources.langsmith as ls_module

    monkeypatch.setattr(ls_module, "LangSmithSource", _BrokenSource)
    rc = main(["fetch", "trace-x", "--anonymizer", "regex", "--data-dir", str(tmp_path)])
    assert rc == 0
    assert "feedback lookup failed" in capsys.readouterr().err
    a = TraceStore(data_root=tmp_path).load_assessment("trace-x")
    assert a is not None
    assert a.outcome == OutcomeStatus.UNKNOWN
    assert "unavailable" in a.notes


# -- LangSmith adapter ----------------------------------------------------------


def test_langsmith_fetch_feedback_keeps_key_and_numeric_score_only():
    from self_improve_cli.sources.langsmith import LangSmithSource

    calls: list[dict] = []

    def list_feedback(**kwargs):
        calls.append(kwargs)
        return [
            SimpleNamespace(key="correctness", score=True, comment="secret text", value=None),
            SimpleNamespace(key="helpfulness", score=0.5, comment=None, value=None),
            SimpleNamespace(key="label", score=None, comment=None, value="free text"),
        ]

    source = LangSmithSource(project_name="p", client=SimpleNamespace(list_feedback=list_feedback))
    assert source.fetch_feedback("trace-x") == [
        FeedbackScore("correctness", 1.0),
        FeedbackScore("helpfulness", 0.5),
        FeedbackScore("label", None),
    ]
    assert calls[0]["run_ids"] == ["trace-x"]
