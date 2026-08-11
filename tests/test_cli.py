"""Tests for the CLI (offline, no network)."""

import json

import pytest

from self_improve_cli.cli.main import build_parser, main
from self_improve_cli.domain import Message, RunType, ToolCall, Trace
from self_improve_cli.storage import TraceStore
from tests.helpers import make_msg, make_root, make_run


@pytest.fixture
def saved_trace(tmp_path, monkeypatch):
    """Save a sanitized trace under a temp data dir."""
    trace = Trace(
        trace_id="trace-1",
        runs=[
            make_root(trace_id="trace-1"),
            make_run(
                "llm-1",
                run_type=RunType.LLM,
                name="ChatOpenAI",
                parent="root",
                trace_id="trace-1",
                total_tokens=50,
                prompt_tokens=40,
                input_messages=[
                    make_msg("system", "You are a helpful agent."),
                    make_msg("human", "What is 2+2?"),
                ],
                output_message=Message(
                    role="ai",
                    text="",
                    tool_calls=[
                        ToolCall(name="calculator", args={"expression": "2+2"}, id="call-1")
                    ],
                ),
            ),
            make_run(
                "tool-1",
                run_type=RunType.TOOL,
                name="calculator",
                parent="root",
                trace_id="trace-1",
                inputs={"expression": "2+2"},
                outputs={"output": {"update": {"messages": [{"content": "4", "type": "tool"}]}}},
            ),
            make_run(
                "llm-2",
                run_type=RunType.LLM,
                name="ChatOpenAI",
                parent="root",
                trace_id="trace-1",
                total_tokens=50,
                prompt_tokens=45,
                input_messages=[
                    make_msg("system", "You are a helpful agent."),
                    make_msg("human", "What is 2+2?"),
                    Message(
                        role="ai",
                        text="",
                        tool_calls=[
                            ToolCall(name="calculator", args={"expression": "2+2"}, id="call-1")
                        ],
                    ),
                    Message(role="tool", text="4", tool_call_id="call-1"),
                ],
                output_message=Message(role="ai", text="2+2 = 4."),
            ),
        ],
        sanitized=True,
        sanitization_report={
            "entity_counts": {},
            "complete": True,
            "recognizer_version": "regex-fallback",
        },
        source="test",
    )
    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(trace)
    return trace.trace_id, tmp_path


def test_skeleton_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["skeleton", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "[llm]" in out


def test_narrative_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["narrative", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "## Task" in out
    assert "2+2 = 4." in out


def test_run_detail_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["run-detail", trace_id, "run-llm-2", "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "You are a helpful agent." in out


def test_tool_metrics_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["tool-metrics", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "# Tool Metrics" in out


def test_context_metrics_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["context-metrics", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "# Context Metrics" in out


def test_info_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["info", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["trace_id"] == trace_id
    assert data["sanitized"] is True


def test_info_command_json_format(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["info", trace_id, "--data-dir", str(data_dir), "--format", "json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["sanitized"] is True


def test_missing_trace_returns_error(saved_trace, capsys):
    _, data_dir = saved_trace
    assert main(["narrative", "unknown-trace", "--data-dir", str(data_dir)]) == 1
    err = capsys.readouterr().err
    assert "fetch" in err


def test_help_mentions_privacy():
    parser = build_parser()
    assert "anonymized" in parser.epilog.lower() or "privacy" in parser.epilog.lower()
