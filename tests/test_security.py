"""Tests for security fixes: path traversal, raw fallback, validation."""

import json

import pytest

from self_improve_cli.domain import Trace
from self_improve_cli.storage import TraceStore, _validate_id
from tests.helpers import make_root


def _make_sanitized_trace(trace_id: str = "t1") -> Trace:
    return Trace(
        trace_id=trace_id,
        runs=[make_root(trace_id=trace_id)],
        sanitized=True,
        sanitization_report={"entity_counts": {}, "complete": True},
        source="test",
    )


# --- Path traversal ---


def test_validate_id_rejects_dotdot():
    with pytest.raises(ValueError, match="Invalid"):
        _validate_id("../../etc")


def test_validate_id_rejects_absolute_path():
    with pytest.raises(ValueError, match="Invalid"):
        _validate_id("C:/Windows/System32")


def test_validate_id_rejects_slash():
    with pytest.raises(ValueError, match="Invalid"):
        _validate_id("foo/bar")


def test_validate_id_rejects_empty():
    with pytest.raises(ValueError, match="Invalid"):
        _validate_id("")


def test_validate_id_accepts_safe_ids():
    assert _validate_id("trace-abc-123") == "trace-abc-123"
    assert _validate_id("abc123") == "abc123"
    assert _validate_id("a.b.c") == "a.b.c"
    assert _validate_id("a_b-c.d") == "a_b-c.d"


def test_save_sanitized_rejects_path_traversal(tmp_path):
    store = TraceStore(data_root=tmp_path)
    trace = _make_sanitized_trace()
    trace.trace_id = "../../etc"
    with pytest.raises(ValueError, match="Invalid trace_id"):
        store.save_sanitized(trace)


def test_save_ter_file_rejects_path_traversal_filename(tmp_path):
    store = TraceStore(data_root=tmp_path)
    with pytest.raises(ValueError, match="Invalid filename"):
        store.save_ter_file("t1", "../../etc/passwd", "content")


def test_save_ter_file_rejects_path_traversal_trace_id(tmp_path):
    store = TraceStore(data_root=tmp_path)
    with pytest.raises(ValueError, match="Invalid trace_id"):
        store.save_ter_file("../../etc", "skeleton.md", "content")


def test_load_trace_rejects_path_traversal(tmp_path):
    store = TraceStore(data_root=tmp_path)
    with pytest.raises(ValueError, match="Invalid trace_id"):
        store.load_trace("../../etc")


# --- Raw fallback ---


def test_load_trace_raises_for_unsanitized_raw(tmp_path):
    """If only raw.json exists and it's not sanitized, raise ValueError."""
    store = TraceStore(data_root=tmp_path, keep_raw=True)
    # Manually create a raw.json with sanitized=False
    trace_dir = tmp_path / "traces" / "t1"
    trace_dir.mkdir(parents=True)
    (trace_dir / "raw.json").write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "runs": [],
                "schema_version": 1,
                "sanitized": False,
                "source": "test",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="NOT sanitized"):
        store.load_trace("t1")


def test_load_trace_works_for_sanitized(tmp_path):
    """Normal sanitized load still works."""
    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(_make_sanitized_trace())
    trace = store.load_trace("t1")
    assert trace.sanitized is True
