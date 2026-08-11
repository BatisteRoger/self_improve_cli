"""Tests for context metrics."""

from self_improve_cli.domain import Message, ToolCall
from self_improve_cli.metrics.context_metrics import (
    build_context_metrics,
    dead_context_ratio,
    growth_curve,
    token_decomposition,
)
from self_improve_cli.representations import significant_runs
from tests.helpers import make_llm, make_msg, make_nested_llm, make_root, make_tool


def test_token_decomposition_empty_trace():
    assert token_decomposition([]) == []


def test_nested_llms_excluded_from_main_loop():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=1000),
        make_tool("2", "evaluate_song"),
        make_nested_llm("3", prompt_tokens=500, tool_id="run-2"),
        make_llm("4", prompt_tokens=3000),
    ]
    sig = significant_runs(runs)
    from self_improve_cli.metrics.context_metrics import _main_loop_llm_runs

    main_llms = _main_loop_llm_runs(sig)
    ids = [r.id for r in main_llms]
    assert "run-1" in ids
    assert "run-4" in ids
    assert "run-3" not in ids


def test_growth_curve_excludes_nested_llms():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=10000),
        make_tool("2", "evaluate_song"),
        make_nested_llm("3", prompt_tokens=500, tool_id="run-2"),
        make_llm("4", prompt_tokens=12000),
    ]
    curve = growth_curve(runs)
    assert len(curve["steps"]) == 1
    assert curve["steps"][0]["delta"] == 2000
    assert not curve["steps"][0]["is_drop"]
    assert curve["is_monotonic"]


def test_growth_curve_detects_jump():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=1000),
        make_tool("2", "big_tool"),
        make_llm("3", prompt_tokens=8000),
    ]
    curve = growth_curve(runs)
    assert curve["steps"][0]["is_jump"]
    assert not curve["steps"][0]["is_drop"]


def test_growth_curve_detects_drop():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=10000),
        make_llm("2", prompt_tokens=5000),
    ]
    curve = growth_curve(runs)
    assert curve["steps"][0]["is_drop"]
    assert not curve["is_monotonic"]


def test_growth_curve_monotonic_no_drops():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=1000),
        make_llm("2", prompt_tokens=2000),
        make_llm("3", prompt_tokens=3000),
    ]
    curve = growth_curve(runs)
    assert curve["is_monotonic"]
    assert not curve["has_drops"]


def test_growth_curve_uses_prompt_tokens_not_total():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=1000),
        make_llm("2", prompt_tokens=2000),
    ]
    curve = growth_curve(runs)
    assert curve["steps"][0]["delta"] == 1000


def test_growth_curve_empty():
    assert growth_curve([])["steps"] == []


def test_dead_context_ratio_empty_trace():
    assert dead_context_ratio([]) == []


def test_dead_context_ratio_counts_tool_messages():
    runs = [
        make_root(),
        make_llm(
            "1",
            prompt_tokens=40,
            input_messages=[
                make_msg("system", "You are a helpful agent."),
                make_msg("human", "What is 2+2?"),
            ],
        ),
        make_llm(
            "2",
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
        ),
    ]
    ratios = dead_context_ratio(runs)
    assert len(ratios) == 2
    assert ratios[0]["total_tool_msgs"] == 0
    assert ratios[1]["total_tool_msgs"] == 1
    assert ratios[1]["ratio"] == 0.0


def test_dead_context_ratio_marks_old_tool_results():
    msgs = []
    for n in range(7):
        msgs.append(
            Message(
                role="ai", text="", tool_calls=[ToolCall(name="read_file", args={}, id=f"c{n}")]
            )
        )
        msgs.append(Message(role="tool", text="x" * 40, tool_call_id=f"c{n}"))
    llm = make_llm("1", prompt_tokens=5000, input_messages=msgs)
    ratios = dead_context_ratio([make_root(), llm])
    assert ratios[0]["total_tool_msgs"] == 7
    assert ratios[0]["dead_tool_msgs"] == 2


def test_build_context_metrics_empty_trace():
    assert "(empty trace)" in build_context_metrics([])


def test_build_context_metrics_contains_sections():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=1000),
        make_llm("2", prompt_tokens=2000),
    ]
    md = build_context_metrics(runs)
    assert "# Context Metrics" in md
    assert "## Growth curve" in md
