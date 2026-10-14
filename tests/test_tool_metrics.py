"""Tests for tool metrics."""

from self_improve_cli.domain import Run
from self_improve_cli.metrics.tool_metrics import (
    build_tool_metrics,
    extract_target_key,
    latency_decomposition,
    modify_granularity,
    process_and_friction_signals,
    repeated_same_target,
    token_cost_attribution,
    tool_call_counts,
    tool_metrics_data,
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


def test_latency_decomposition_merges_parallel_tool_intervals():
    root = make_root()
    root.start_time = "2026-10-14T10:00:00Z"
    root.end_time = "2026-10-14T10:00:10Z"

    llm1 = make_llm("1", prompt_tokens=100)
    llm1.start_time = "2026-10-14T10:00:00Z"
    llm1.end_time = "2026-10-14T10:00:02Z"  # 2.0s

    # Two parallel tool calls running from t=2s to t=6s (4s wall-clock each, overlapped)
    t1 = make_tool("2", "web_search", {"query": "a"})
    t1.start_time = "2026-10-14T10:00:02Z"
    t1.end_time = "2026-10-14T10:00:06Z"

    t2 = make_tool("3", "web_search", {"query": "b"})
    t2.start_time = "2026-10-14T10:00:03Z"
    t2.end_time = "2026-10-14T10:00:07Z"
    t2.error = "HTTP 500"
    t2.status = "error"

    llm2 = make_llm("4", prompt_tokens=200)
    llm2.start_time = "2026-10-14T10:00:08Z"
    llm2.end_time = "2026-10-14T10:00:10Z"  # 2.0s (1s gap between t=7s and t=8s)

    lat = latency_decomposition([root, llm1, t1, t2, llm2])
    assert lat["available"] is True
    assert lat["total_s"] == 10.0
    assert lat["llm_time_s"] == 4.0
    # Merged tool window is [2s, 7s] = 5.0s, NOT 4s + 4s = 8s
    assert lat["tool_time_s"] == 5.0
    assert lat["failed_tool_time_s"] == 4.0
    assert lat["unexplained_gap_s"] == 1.0
    assert lat["per_tool"]["web_search"]["calls"] == 2
    assert lat["per_tool"]["web_search"]["total_s"] == 8.0


def test_process_and_friction_signals_retries_oscillation_and_oversized():
    # 1. Identical failed retry on edit_file
    t1 = make_tool("1", "edit_file", {"path": "a.py", "old_string": "x"})
    t1.error = "old_string not found"
    t1.status = "error"
    t2 = make_tool("2", "edit_file", {"path": "a.py", "old_string": "x"})
    t2.error = "old_string not found"
    t2.status = "error"
    # 2. Adapted retry after error that succeeds
    t3 = make_tool("3", "edit_file", {"path": "a.py", "old_string": "y"})

    # 3. Oscillation A -> B -> A -> B on read_file
    o1 = make_tool("4", "read_file", {"path": "foo.py"})
    o2 = make_tool("5", "read_file", {"path": "bar.py"})
    o3 = make_tool("6", "read_file", {"path": "foo.py"})
    o4 = make_tool("7", "read_file", {"path": "bar.py"})
    o4.outputs = {"output": "z" * 10000}  # 2500 estimated tokens -> oversized

    runs = [make_root(), t1, t2, t3, o1, o2, o3, o4]
    sig = process_and_friction_signals(runs, oversized_tokens=2000)

    assert len(sig["failed_retries"]) == 2
    assert sig["failed_retries"][0]["mode"] == "identical_args"
    assert sig["failed_retries"][0]["failed_run_id"] == "run-1"
    assert sig["failed_retries"][0]["retry_run_id"] == "run-2"
    assert sig["failed_retries"][0]["suggested_pattern"] == "recovery.failed_recovery"

    assert sig["failed_retries"][1]["mode"] == "adapted_after_error"
    assert sig["failed_retries"][1]["failed_run_id"] == "run-2"
    assert sig["failed_retries"][1]["retry_run_id"] == "run-3"
    assert sig["failed_retries"][1]["suggested_pattern"] == "tool.probabilistic_discovery"

    assert len(sig["oscillations"]) == 1
    osc = sig["oscillations"][0]
    assert osc["length"] == 4
    assert osc["run_ids"] == ["run-4", "run-5", "run-6", "run-7"]
    assert osc["suggested_pattern"] == "control.nonprogress_loop"

    assert len(sig["oversized_outputs"]) == 1
    ov = sig["oversized_outputs"][0]
    assert ov["run_id"] == "run-7"
    assert ov["estimated_tokens"] >= 2500
    assert ov["suggested_candidate"] == "harness.observation_pack"

    md = build_tool_metrics(runs)
    assert "## Process & friction signals" in md
    assert "Failed retry [identical_args]" in md
    assert "Oversized tool output: read_file (bar.py) id=run-7" in md
    # Oscillations and suggested pattern labels are JSON-only.
    assert "Oscillation" not in md
    assert "candidate" not in md

    data = tool_metrics_data(runs)
    assert "process_and_friction_signals" in data


def _failed_tool(run_id: str, name: str, args: dict) -> Run:
    run = make_tool(run_id, name, args)
    run.error = "not found"
    run.status = "error"
    return run


def test_failed_retry_requires_same_target():
    """A same-name call on a different target after a failure is not a retry."""
    runs = [
        make_root(),
        _failed_tool("1", "read_file", {"path": "a.py"}),
        make_tool("2", "read_file", {"path": "b.py"}),
    ]
    assert process_and_friction_signals(runs)["failed_retries"] == []


def test_failed_retry_requires_window():
    """A same-target call more than 5 tool calls after the failure is not a retry."""
    runs = [make_root(), _failed_tool("1", "read_file", {"path": "a.py", "limit": 10})]
    runs += [make_tool(f"{i}", "list_dir", {"path": f"d{i}"}) for i in range(2, 7)]
    runs.append(make_tool("7", "read_file", {"path": "a.py", "limit": 20}))
    assert process_and_friction_signals(runs)["failed_retries"] == []

    within = runs[:3] + [runs[-1]]
    retries = process_and_friction_signals(within)["failed_retries"]
    assert [r["mode"] for r in retries] == ["adapted_after_error"]


def test_oscillation_requires_two_cycles():
    """A -> B -> A (3 calls) is not an oscillation; 4 calls are needed."""
    runs = [
        make_root(),
        make_tool("1", "read_file", {"path": "foo.py"}),
        make_tool("2", "read_file", {"path": "bar.py"}),
        make_tool("3", "read_file", {"path": "foo.py"}),
    ]
    assert process_and_friction_signals(runs)["oscillations"] == []
