"""Tests for tool metrics."""

from self_improve_cli.metrics.tool_metrics import (
    build_tool_metrics,
    extract_target_key,
    modify_granularity,
    repeated_same_target,
    token_cost_attribution,
    tool_call_counts,
)
from tests.helpers import make_llm, make_root, make_tool


def test_extract_target_key_prefers_path():
    assert extract_target_key("read_file", {"path": "lyrics.md", "limit": 10}) == "lyrics.md"


def test_extract_target_key_fallback_short_identifier():
    assert extract_target_key("calculator", {"expression": "2+2"}) == "2+2"


def test_extract_target_key_rejects_prose():
    args = {"query": "what is the meaning of life and everything else"}
    assert extract_target_key("search", args) is None


def test_extract_target_key_ignores_internal_keys():
    assert extract_target_key("foo", {"tool_call_id": "abc123"}) is None


def test_tool_call_counts_empty():
    assert tool_call_counts([]) == {}


def test_repeated_same_target_detects_repeats():
    runs = [
        make_root(),
        make_tool("1", "edit_file", {"path": "a.md", "old_string": "x", "new_string": "y"}),
        make_tool("2", "edit_file", {"path": "a.md", "old_string": "p", "new_string": "q"}),
        make_tool("3", "edit_file", {"path": "a.md", "old_string": "m", "new_string": "n"}),
        make_tool("4", "edit_file", {"path": "b.md", "old_string": "1", "new_string": "2"}),
    ]
    result = repeated_same_target(runs, threshold=3)
    assert result == {"edit_file": [("a.md", 3)]}


def test_repeated_same_target_below_threshold_excluded():
    runs = [make_root(), make_tool("1", "read_file", {"path": "a.md"})]
    assert repeated_same_target(runs, threshold=3) == {}


def test_modify_granularity_counts_lines():
    runs = [
        make_root(),
        make_tool(
            "1",
            "edit_file",
            {"path": "a.md", "old_string": "one\ntwo", "new_string": "ONE\nTWO\nTHREE"},
        ),
        make_tool("2", "edit_file", {"path": "a.md", "old_string": "x", "new_string": "y"}),
    ]
    gran = modify_granularity(runs)
    assert gran["total_modify_calls"] == 2
    assert gran["total_units_changed"] == 4
    assert gran["modify_efficiency"] == 2.0


def test_modify_tool_detected_by_arg_hint():
    runs = [
        make_root(),
        make_tool("1", "apply_diff", {"target": "a", "old_string": "x", "new_string": "y"}),
    ]
    assert modify_granularity(runs)["total_modify_calls"] == 1


def test_token_cost_attribution_uses_prompt_tokens():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=1000),
        make_tool("2", "evaluate", {"slug": "song"}),
        make_llm("3", prompt_tokens=5000),
        make_tool("4", "read_file", {"path": "a.md"}),
        make_llm("5", prompt_tokens=6000),
    ]
    attrib = token_cost_attribution(runs)
    assert attrib["evaluate"]["total_delta"] == 4000
    assert attrib["read_file"]["total_delta"] == 1000
    assert list(attrib.keys())[0] == "evaluate"


def test_token_cost_attribution_splits_between_concurrent_tools():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=1000),
        make_tool("2", "tool_a", {"path": "a"}),
        make_tool("3", "tool_b", {"path": "b"}),
        make_llm("4", prompt_tokens=3000),
    ]
    attrib = token_cost_attribution(runs)
    assert attrib["tool_a"]["total_delta"] == 1000
    assert attrib["tool_b"]["total_delta"] == 1000


def test_token_cost_attribution_ignores_negative_deltas():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=5000),
        make_tool("2", "compact", {"path": "x"}),
        make_llm("3", prompt_tokens=2000),
    ]
    assert token_cost_attribution(runs) == {}


def test_build_tool_metrics_empty_trace():
    assert "(empty trace)" in build_tool_metrics([])
