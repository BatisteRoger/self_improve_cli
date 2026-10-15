"""Tests for the trajectory diff view and the `diff` CLI command (offline)."""

import json

import pytest

from self_improve_cli.cli.main import main
from self_improve_cli.domain import RunType, ToolCall, Trace
from self_improve_cli.representations import build_trace_diff, trace_diff_data
from self_improve_cli.storage import TraceStore
from tests.helpers import make_msg, make_root, make_run


def _llm(idx: str, trace_id: str, calls: list[tuple[str, dict]] | None = None):
    out = make_msg("ai", "ok", tool_calls=[ToolCall(name=n, args=a) for n, a in calls or []])
    return make_run(idx, RunType.LLM, prompt_tokens=100, trace_id=trace_id, output_message=out)


def _tool(idx: str, name: str, args: dict, trace_id: str, error: str | None = None):
    return make_run(idx, RunType.TOOL, name=name, inputs=args, trace_id=trace_id, error=error)


def _runs_a() -> list:
    t = "trace-a"
    return [
        make_root(trace_id=t),
        _llm("a1", t, [("search", {"q": "x"})]),
        _tool("a2", "search", {"q": "x"}, t),
        _llm("a3", t, [("read_file", {"path": "f"})]),
        _tool("a4", "read_file", {"path": "f"}, t),
        _llm("a5", t),
    ]


def _runs_b() -> list:
    t = "trace-b"
    return [
        make_root(trace_id=t),
        _llm("b1", t, [("search", {"q": "y"})]),
        _tool("b2", "search", {"q": "y"}, t, error="HTTP 500"),
        _llm("b3", t, [("write_file", {"path": "g"})]),
        _tool("b4", "write_file", {"path": "g"}, t),
        _tool("b5", "write_file", {"path": "h"}, t),
        _llm("b6", t),
    ]


def test_diff_data_alignment():
    data = trace_diff_data(_runs_a(), _runs_b(), "trace-a", "trace-b")
    s = data["summary"]
    assert (s["steps_a"], s["steps_b"]) == (5, 6)
    assert s["aligned"] == 4  # llm, search, llm, final llm
    assert s["only_in_a"] == 1 and s["only_in_b"] == 2
    assert s["identical"] is False

    search = next(
        seg
        for seg in data["segments"]
        if seg.get("a", {}) and seg["kind"] == "aligned" and seg["a"]["name"] == "search"
    )
    assert search["a"]["run_id"] == "run-a2" and search["b"]["run_id"] == "run-b2"
    assert search["a"]["step"] == 2 and search["b"]["step"] == 2
    assert set(search["changes"]) == {"error", "args"}
    assert search["changes"]["error"] == {"a": None, "b": "HTTP 500"}

    first_llm = data["segments"][0]
    assert "tool_calls" in first_llm["changes"]
    assert first_llm["token_delta"] == 0

    div = next(seg for seg in data["segments"] if seg["kind"] == "divergence")
    assert [r["run_id"] for r in div["a"]] == ["run-a4"]
    assert [r["run_id"] for r in div["b"]] == ["run-b4", "run-b5"]
    assert [r["run_id"] for r in data["only_in_b"]] == ["run-b4", "run-b5"]

    last = data["segments"][-1]
    assert last["kind"] == "aligned"
    assert (last["a"]["step"], last["b"]["step"]) == (5, 6)
    assert last["changes"] == {}


def test_diff_markdown_cites_both_sides():
    out = build_trace_diff(_runs_a(), _runs_b(), "trace-a", "trace-b")
    assert "Steps A=5 B=6: 4 aligned (3 with changes)" in out
    assert "A#2 = B#2 [tool] search — changed (ids A=run-a2 B=run-b2" in out
    assert "error: A=None | B=HTTP 500" in out
    assert "### Divergence A#4-4 vs B#4-5" in out
    assert "B#5 [tool] write_file id=run-b5" in out
    assert "A#5-5 = B#6-6: 1 aligned step(s), unchanged" in out
    assert "run-detail trace-a <run_id>" in out


def test_diff_identical_trajectories():
    runs_b = [make_root(trace_id="trace-b")] + [
        _llm("b1", "trace-b", [("search", {"q": "x"})]),
        _tool("b2", "search", {"q": "x"}, "trace-b"),
    ]
    runs_a = _runs_a()[:3]
    data = trace_diff_data(runs_a, runs_b, "trace-a", "trace-b")
    assert data["summary"]["identical"] is True
    out = build_trace_diff(runs_a, runs_b, "trace-a", "trace-b")
    assert "No divergence" in out
    assert "A#1-2 = B#1-2: 2 aligned step(s), unchanged" in out
    full = build_trace_diff(runs_a, runs_b, "trace-a", "trace-b", full=True)
    assert "A#1 = B#1 [llm] ChatModel — unchanged" in full


def test_diff_long_divergence_is_bounded():
    t = "trace-b"
    runs_b = [make_root(trace_id=t)] + [_tool(f"b{i:02d}", f"t{i}", {}, t) for i in range(15)]
    runs_a = [make_root(trace_id="trace-a"), _llm("a1", "trace-a")]
    out = build_trace_diff(runs_a, runs_b, "trace-a", "trace-b")
    assert "[omitted 5 B step(s)" in out
    assert "narrative trace-b --from 11 --to 15" in out
    assert "omitted" not in build_trace_diff(runs_a, runs_b, "trace-a", "trace-b", full=True)


def test_diff_long_args_truncated():
    big = {"q": "z" * 500}
    runs_a = [make_root("trace-a"), _tool("a1", "search", {"q": "x"}, "trace-a")]
    runs_b = [make_root("trace-b"), _tool("b1", "search", big, "trace-b")]
    data = trace_diff_data(runs_a, runs_b, "trace-a", "trace-b")
    assert "[truncated," in data["segments"][0]["changes"]["args"]["b"]


def test_diff_root_name_mismatch_noted():
    runs_b = _runs_b()
    runs_b[0].name = "other-agent"
    data = trace_diff_data(_runs_a(), runs_b, "trace-a", "trace-b")
    assert data["notes"] and "different agents" in data["notes"][0]


def test_diff_same_id_raises():
    with pytest.raises(ValueError, match="two different trace IDs"):
        trace_diff_data(_runs_a(), _runs_a(), "trace-a", "trace-a")


def test_diff_empty_trace_raises():
    with pytest.raises(ValueError, match="skeleton trace-b"):
        trace_diff_data(_runs_a(), [make_root("trace-b")], "trace-a", "trace-b")


# --- CLI ---------------------------------------------------------------------


@pytest.fixture
def stored(tmp_path):
    store = TraceStore(data_root=tmp_path)
    for tid, runs in (
        ("trace-a", _runs_a()),
        ("trace-b", _runs_b()),
        ("trace-empty", [make_root("trace-empty")]),
    ):
        store.save_sanitized(
            Trace(
                trace_id=tid,
                runs=runs,
                sanitized=True,
                sanitization_report={"entity_counts": {}, "complete": True},
                source="test",
            )
        )
    return str(tmp_path)


def test_cli_diff_markdown(stored, capsys):
    assert main(["diff", "trace-a", "trace-b", "--data-dir", stored]) == 0
    out = capsys.readouterr().out
    assert "# Trajectory diff — A=trace-a vs B=trace-b" in out
    assert "id=run-b5" in out


def test_cli_diff_json(stored, capsys):
    assert main(["diff", "trace-a", "trace-b", "--data-dir", stored, "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["trace_a"] == "trace-a" and data["trace_b"] == "trace-b"
    assert {"segments", "only_in_a", "only_in_b", "summary"} <= set(data)
    assert data["summary"]["aligned"] == 4


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["trace-a", "missing"], "self-improve fetch"),
        (["trace-a", "trace-a"], "two different trace IDs"),
        (["trace-a", "trace-empty"], "skeleton trace-empty"),
    ],
)
def test_cli_diff_errors(stored, capsys, argv, expected):
    assert main(["diff", *argv, "--data-dir", stored]) == 1
    assert expected in capsys.readouterr().err
