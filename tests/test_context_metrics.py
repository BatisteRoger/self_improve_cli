"""Tests for context metrics."""

from self_improve_cli.domain import Message, ToolCall
from self_improve_cli.metrics.context_metrics import (
    build_context_metrics,
    context_metrics_data,
    growth_curve,
    prefix_invariance,
    stale_tool_result_ratio,
    token_decomposition,
    token_lifecycle,
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


def test_stale_tool_result_ratio_empty_trace():
    assert stale_tool_result_ratio([]) == []


def test_stale_tool_result_ratio_counts_tool_messages():
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
    ratios = stale_tool_result_ratio(runs)
    assert len(ratios) == 2
    assert ratios[0]["total_tool_msgs"] == 0
    assert ratios[1]["total_tool_msgs"] == 1
    assert ratios[1]["ratio"] == 0.0


def test_stale_tool_result_ratio_marks_old_tool_results():
    msgs = []
    for n in range(7):
        msgs.append(
            Message(
                role="ai", text="", tool_calls=[ToolCall(name="read_file", args={}, id=f"c{n}")]
            )
        )
        msgs.append(Message(role="tool", text="x" * 40, tool_call_id=f"c{n}"))
    llm = make_llm("1", prompt_tokens=5000, input_messages=msgs)
    ratios = stale_tool_result_ratio([make_root(), llm])
    assert ratios[0]["total_tool_msgs"] == 7
    assert ratios[0]["stale_tool_msgs"] == 2


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


def test_prefix_invariance_empty_and_single_step():
    assert prefix_invariance([]) == {"steps": [], "median_share": None, "repaid_tokens": 0}
    runs = [make_root(), make_llm("1", prompt_tokens=100)]
    inv = prefix_invariance(runs)
    assert len(inv["steps"]) == 1
    assert inv["steps"][0]["share"] is None
    assert inv["median_share"] is None


def test_prefix_invariance_shared_prefix():
    runs = [
        make_root(),
        make_llm(
            "1",
            prompt_tokens=100,
            input_messages=[make_msg("system", "SYS"), make_msg("human", "Q")],
        ),
        make_llm(
            "2",
            prompt_tokens=200,
            input_messages=[
                make_msg("system", "SYS"),
                make_msg("human", "Q"),
                make_msg("ai", "answer"),
                make_msg("human", "Q2"),
            ],
        ),
    ]
    inv = prefix_invariance(runs)
    step1 = inv["steps"][1]
    expected_chars = len("system\nSYS\n\nhuman\nQ")
    assert step1["shared_prefix_chars"] == expected_chars
    assert step1["shared_prefix_tokens"] == expected_chars // 4
    assert step1["share"] == round((expected_chars // 4) / 200, 2)
    assert inv["repaid_tokens"] == expected_chars // 4
    assert inv["median_share"] == step1["share"]


def test_prefix_invariance_mid_message_divergence():
    runs = [
        make_root(),
        make_llm(
            "1",
            prompt_tokens=400,
            input_messages=[make_msg("system", "SYS user=alice tail")],
        ),
        make_llm(
            "2",
            prompt_tokens=400,
            input_messages=[make_msg("system", "SYS user=bob tail")],
        ),
    ]
    inv = prefix_invariance(runs)
    step1 = inv["steps"][1]
    # Char-level keeps the common head of the system message; message-level
    # matching would report zero here.
    assert step1["shared_prefix_chars"] == len("system\nSYS user=")
    assert step1["shared_prefix_tokens"] > 0


def test_prefix_invariance_rebuilt_context():
    runs = [
        make_root(),
        make_llm(
            "1",
            prompt_tokens=400,
            input_messages=[make_msg("system", "AAA"), make_msg("human", "one")],
        ),
        make_llm(
            "2",
            prompt_tokens=400,
            input_messages=[make_msg("system", "ZZZ"), make_msg("human", "two")],
        ),
    ]
    inv = prefix_invariance(runs)
    assert inv["steps"][1]["shared_prefix_chars"] == len("system\n")
    assert inv["steps"][1]["share"] == 0.0


def test_context_metrics_data_has_prefix_invariance():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=100, input_messages=[make_msg("system", "SYS")]),
        make_llm("2", prompt_tokens=200, input_messages=[make_msg("system", "SYS")]),
    ]
    data = context_metrics_data(runs)
    assert "prefix_invariance" in data
    assert len(data["prefix_invariance"]["steps"]) == 2
    assert "## Static context" in build_context_metrics(runs)


def test_token_lifecycle_empty_trace():
    assert token_lifecycle([]) == {
        "items": [],
        "duplicate_tool_outputs": [],
        "churn_events": [],
    }


def test_token_lifecycle_tracks_repeated_exposure_and_drop():
    big_tool_text = "x" * 1600  # 400 tokens (chars/4)
    sys_msg = make_msg("system", "s" * 1200)  # 300 tokens
    tc_msg = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="read_file", args={"path": "a.py"}, id="call-1")],
    )
    tool_msg = Message(role="tool", text=big_tool_text, tool_call_id="call-1")

    runs = [
        make_root(),
        make_llm("1", prompt_tokens=300, input_messages=[sys_msg]),
        make_llm("2", prompt_tokens=700, input_messages=[sys_msg, tc_msg, tool_msg]),
        make_llm("3", prompt_tokens=800, input_messages=[sys_msg, tc_msg, tool_msg]),
        # Step 3 (0-based): compaction drops tool_msg
        make_llm("4", prompt_tokens=350, input_messages=[sys_msg]),
    ]
    lc = token_lifecycle(runs, min_tokens=250)
    by_source = {item["source"]: item for item in lc["items"]}

    read_item = by_source["tool:read_file"]
    assert read_item["tool_call_id"] == "call-1"
    assert read_item["introduced_step"] == 1
    assert read_item["last_step"] == 2
    assert read_item["steps_present"] == 2
    assert read_item["dropped_at_step"] == 3
    assert read_item["tokens"] == 400
    assert read_item["cumulative_tokens"] == 800
    assert read_item["repaid_tokens"] == 400

    sys_item = by_source["system"]
    assert sys_item["introduced_step"] == 0
    assert sys_item["last_step"] == 3
    assert sys_item["steps_present"] == 4
    assert sys_item["dropped_at_step"] is None
    assert sys_item["repaid_tokens"] == 900

    md = build_context_metrics(runs)
    assert "## Token-cost lifecycle (approximate)" in md
    assert "tool:read_file (call-1)" in md
    assert "[dropped at step 3]" in md


def test_token_lifecycle_detects_duplicate_tool_outputs_and_churn():
    payload = "content-" * 40  # 320 chars -> 80 tokens (>= _LIFECYCLE_DUP_MIN_TOKENS)
    tc1 = Message(
        role="ai",
        text="",
        tool_calls=[
            ToolCall(name="read_file", args={"path": "a.py"}, id="call-1"),
            ToolCall(name="read_file", args={"path": "a.py"}, id="call-2"),
        ],
    )
    t1 = Message(role="tool", text=payload, tool_call_id="call-1")
    t2 = Message(role="tool", text=payload, tool_call_id="call-2")
    tc3 = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="read_file", args={"path": "a.py"}, id="call-3")],
    )
    t3 = Message(role="tool", text=payload, tool_call_id="call-3")

    runs = [
        make_root(),
        # Step 0: two identical tool outputs in the same context
        make_llm("1", prompt_tokens=500, input_messages=[tc1, t1, t2]),
        # Step 1: compacted away
        make_llm("2", prompt_tokens=200, input_messages=[make_msg("system", "summary")]),
        # Step 2: re-fetched the exact same payload
        make_llm("3", prompt_tokens=400, input_messages=[tc3, t3]),
    ]
    lc = token_lifecycle(runs)
    assert len(lc["duplicate_tool_outputs"]) == 1
    dup = lc["duplicate_tool_outputs"][0]
    assert dup["tool"] == "read_file"
    assert dup["first_step"] == 0
    assert dup["copies"] == 2
    assert dup["duplicate_tokens"] == 80
    assert dup["tool_call_ids"] == ["call-1", "call-2"]

    assert len(lc["churn_events"]) == 1
    churn = lc["churn_events"][0]
    assert churn["source"] == "tool:read_file"
    assert churn["dropped_at_step"] == 1
    assert churn["reintroduced_at_step"] == 2

    data = context_metrics_data(runs)
    assert "token_lifecycle" in data
    md = build_context_metrics(runs)
    assert "Duplicate tool output at step 0: read_file x2" in md
    assert "Context churn: tool:read_file" in md
