"""CLI: `fetch --experiment` and `gate --baseline/--candidate` (offline, synthetic)."""

from __future__ import annotations

import json

import pytest

from self_improve_cli.cli.main import main
from self_improve_cli.domain import (
    ExperimentListing,
    ExperimentRunRef,
    FeedbackScore,
    RunType,
    Trace,
)
from self_improve_cli.storage import TraceStore
from tests.helpers import make_run


def _trace(trace_id: str, example_id: str | None = None, model: str = "m1") -> Trace:
    root = make_run(
        "root",
        run_type=RunType.CHAIN,
        name="agent",
        parent=None,
        trace_id=trace_id,
        extra={"reference_example_id": example_id},
    )
    llm = make_run(
        "l1",
        trace_id=trace_id,
        total_tokens=100,
        extra={"metadata": {"ls_model_name": model}},
    )
    return Trace(trace_id=trace_id, runs=[root, llm], sanitized=True, source="test")


def _experiment(store: TraceStore, sid: str, patterns: dict[str, str], policy=None) -> None:
    examples = {}
    for ex, pattern in patterns.items():
        runs = []
        for i, ch in enumerate(pattern):
            tid = f"{sid}-{ex}-{i}"
            store.save_sanitized(_trace(tid, ex))
            store.save_feedback(tid, [FeedbackScore(key="k", score=1.0 if ch == "P" else 0.0)])
            outcome = "success" if ch == "P" else "fail"
            runs.append({"run_id": tid, "trace_id": tid, "status": "success", "outcome": outcome})
        examples[ex] = runs
    store.save_experiment(
        {
            "session_id": sid,
            "experiment_name": f"exp {sid}",
            "dataset_id": "d1",
            "outcome_policy": policy,
            "conditions": {"git_commit": sid},
            "examples": examples,
        }
    )


def _gate(tmp_path, *extra: str) -> int:
    return main(["gate", *extra, "--data-dir", str(tmp_path), "--format", "json"])


@pytest.fixture
def store(tmp_path):
    s = TraceStore(data_root=tmp_path)
    policy = {"key": "k", "threshold": None, "source": "fetch"}
    # baseline fully stable -> floor 0, so the single regression on `a` fails
    _experiment(s, "s-base", {"a": "PPP", "b": "PPP", "c": "PPP"}, policy)
    _experiment(s, "s-cand", {"a": "FFF", "b": "PPP", "c": "PPP"}, policy)
    return s


def test_gate_sets_fail_by_name_and_id(store, tmp_path, capsys):
    assert _gate(tmp_path, "--baseline", "exp s-base", "--candidate", "s-cand") == 1
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "sets" and data["verdict"] == "fail"
    assert data["criteria"]["quality"]["regressions"] == 1
    facts = {f["name"]: f for f in data["criteria"]["conditions"]["facts"]}
    assert facts["models"]["state"] == "same" and facts["git_commit"]["state"] == "differs"


def test_gate_sets_unknown_experiment(store, tmp_path, capsys):
    code = _gate(tmp_path, "--baseline", "s-cand", "--candidate", "s-cand2", "--strict-conditions")
    assert code == 1  # unknown experiment -> error
    assert "fetch --experiment" in capsys.readouterr().err


def test_gate_sets_outcome_key_override(store, tmp_path, capsys):
    code = _gate(tmp_path, "--baseline", "s-base", "--candidate", "s-cand", "--outcome-key", "x")
    assert code == 2
    data = json.loads(capsys.readouterr().out)
    assert data["reason_code"] == "quality_unknown"
    assert data["baseline"]["outcome_policy"]["source"] == "gate"


def test_gate_sets_id_files(store, tmp_path, capsys):
    base = tmp_path / "base.txt"
    base.write_text("# baseline\ns-base-a-0\ns-base-a-1\na s-base-a-2\n", encoding="utf-8")
    cand = tmp_path / "cand.txt"
    cand.write_text("s-cand-a-0\ns-cand-a-1\n", encoding="utf-8")
    assert _gate(tmp_path, "--baseline", f"@{base}", "--candidate", f"@{cand}") == 2
    data = json.loads(capsys.readouterr().out)
    # no assessments stored -> outcomes unknown, never guessed
    assert data["reason_code"] == "quality_unknown"
    assert (
        _gate(tmp_path, "--baseline", f"@{base}", "--candidate", f"@{cand}", "--outcome-key", "k")
        == 1
    )


def test_gate_sets_id_file_without_example(tmp_path, capsys):
    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(_trace("t1", None))
    ids = tmp_path / "ids.txt"
    ids.write_text("t1\n", encoding="utf-8")
    assert _gate(tmp_path, "--baseline", f"@{ids}", "--candidate", f"@{ids}") == 1
    assert "example_id trace_id" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["before", "after", "--baseline", "x", "--candidate", "y"],
        ["--baseline", "x"],
        ["before"],
        ["before", "after", "--strict-conditions"],
    ],
)
def test_gate_usage_errors(tmp_path, capsys, argv):
    assert _gate(tmp_path, *argv) == 2
    assert "Error:" in capsys.readouterr().err


def test_gate_sets_same_set_rejected(store, tmp_path, capsys):
    assert _gate(tmp_path, "--baseline", "s-base", "--candidate", "exp s-base") == 1
    assert "same set" in capsys.readouterr().err


def test_gate_sets_markdown(store, tmp_path, capsys):
    main(["gate", "--baseline", "s-base", "--candidate", "s-cand", "--data-dir", str(tmp_path)])
    assert capsys.readouterr().out.startswith("# Gate (sets): FAIL")


class _FakeSource:
    def __init__(self, project_name=None):
        self.project_name = project_name

    def fetch_experiment(self, name):
        fb = [FeedbackScore(key="k", score=0.9)]
        runs = [
            ExperimentRunRef("r1", "t1", "ex1", "success", fb),
            ExperimentRunRef("r2", "t2", "ex1", "success", []),
            ExperimentRunRef("r3", "t3", "ex2", "error", fb),
        ]
        return ExperimentListing(name, "sess-1", "d1", {"git_commit": None}, runs)

    def fetch_trace(self, trace_id, project_id=None):
        assert project_id == "sess-1"
        if trace_id == "t3":
            raise RuntimeError("boom")
        return _trace(trace_id, "ex1")


def test_fetch_experiment_cli(tmp_path, capsys, monkeypatch):
    import self_improve_cli.sources.langsmith as ls

    monkeypatch.setattr(ls, "LangSmithSource", _FakeSource)
    argv = [
        "fetch", "--experiment", "exp", "--outcome-key", "k>=0.5",
        "--anonymizer", "regex", "--data-dir", str(tmp_path), "--format", "json",
    ]  # fmt: skip
    assert main(argv) == 1  # t3 failed -> incomplete
    out = json.loads(capsys.readouterr().out)
    assert out["traces"] == {"stored": 2, "skipped": 0, "failed": 1}
    assert out["outcomes"] == {"success": 1, "fail": 0, "unknown": 1}
    store = TraceStore(data_root=tmp_path)
    receipt = store.load_experiment("sess-1")
    assert receipt is not None and receipt["complete"] is False
    assert receipt["failures"][0]["run_id"] == "r3"
    assert receipt["outcome_policy"] == {"key": "k", "threshold": 0.5, "source": "fetch"}
    assessment = store.load_assessment("t1")
    assert assessment is not None and assessment.outcome.value == "success"
    # resume: stored traces are skipped
    assert main(argv) == 1
    assert json.loads(capsys.readouterr().out)["traces"]["skipped"] == 2


def test_fetch_usage_errors(tmp_path, capsys):
    base = ["--data-dir", str(tmp_path)]
    assert main(["fetch", *base]) == 2
    assert main(["fetch", "t1", "--experiment", "x", *base]) == 2
    assert main(["fetch", "t1", "--outcome-key", "k", *base]) == 2
    assert main(["fetch", "--experiment", "x", "--from-run", *base]) == 2
