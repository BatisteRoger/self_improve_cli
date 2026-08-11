"""Tests for the storage layer."""

import json

import pytest

from self_improve_cli.domain import Message, RunType, ToolCall, Trace
from self_improve_cli.storage import TraceStore
from tests.helpers import make_root, make_run


def _make_sanitized_trace(trace_id: str = "t1") -> Trace:
    return Trace(
        trace_id=trace_id,
        runs=[make_root(trace_id=trace_id)],
        sanitized=True,
        sanitization_report={"entity_counts": {"EMAIL": 1}, "complete": True},
        source="langsmith",
    )


def test_save_sanitized_persists_json(tmp_path):
    store = TraceStore(data_root=tmp_path)
    trace = _make_sanitized_trace()
    path = store.save_sanitized(trace)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["trace_id"] == "t1"
    assert data["sanitized"] is True


def test_save_sanitized_refuses_non_sanitized(tmp_path):
    store = TraceStore(data_root=tmp_path)
    trace = Trace(trace_id="t1", runs=[make_root()], sanitized=False)
    with pytest.raises(ValueError, match="non-sanitized"):
        store.save_sanitized(trace)


def test_save_raw_skipped_by_default(tmp_path):
    store = TraceStore(data_root=tmp_path, keep_raw=False)
    trace = _make_sanitized_trace()
    result = store.save_raw(trace)
    assert result is None
    assert not (tmp_path / "traces" / "t1" / "raw.json").exists()


def test_save_raw_with_opt_in(tmp_path):
    store = TraceStore(data_root=tmp_path, keep_raw=True)
    trace = _make_sanitized_trace()
    path = store.save_raw(trace)
    assert path is not None
    assert path.exists()
    assert path.name == "raw.json"


def test_load_sanitized_trace(tmp_path):
    store = TraceStore(data_root=tmp_path)
    trace = _make_sanitized_trace()
    store.save_sanitized(trace)
    loaded = store.load_trace("t1")
    assert loaded.trace_id == "t1"
    assert loaded.sanitized is True
    assert len(loaded.runs) == 1


def test_load_missing_trace_raises(tmp_path):
    store = TraceStore(data_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load_trace("nonexistent")


def test_trace_exists(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assert not store.trace_exists("t1")
    store.save_sanitized(_make_sanitized_trace())
    assert store.trace_exists("t1")


def test_save_and_load_ter_file(tmp_path):
    store = TraceStore(data_root=tmp_path)
    store.save_ter_file("t1", "skeleton.md", "# Skeleton\n")
    content = store.load_ter_file("t1", "skeleton.md")
    assert content == "# Skeleton\n"


def test_load_ter_file_missing_returns_none(tmp_path):
    store = TraceStore(data_root=tmp_path)
    assert store.load_ter_file("t1", "skeleton.md") is None


def test_round_trip_preserves_messages(tmp_path):
    store = TraceStore(data_root=tmp_path)
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(trace_id="t1"),
            make_run(
                "llm-1",
                run_type=RunType.LLM,
                input_messages=[
                    Message(role="system", text="You are helpful."),
                    Message(role="human", text="Hello"),
                ],
                output_message=Message(
                    role="ai",
                    text="Hi!",
                    tool_calls=[ToolCall(name="greet", args={"x": 1}, id="c1")],
                ),
            ),
        ],
        sanitized=True,
    )
    store.save_sanitized(trace)
    loaded = store.load_trace("t1")
    assert len(loaded.runs) == 2
    assert loaded.runs[1].input_messages[0].text == "You are helpful."
    assert loaded.runs[1].output_message.tool_calls[0].name == "greet"
