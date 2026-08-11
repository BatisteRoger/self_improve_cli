"""Tests for progressive representations (TER)."""

from self_improve_cli.domain import Message, RunType, ToolCall
from self_improve_cli.representations import (
    build_narrative,
    build_skeleton,
    run_detail,
    significant_runs,
)
from tests.helpers import make_msg, make_root, make_run


def _mini_runs():
    """Build canonical runs matching the mini_trace fixture structure."""
    root = make_root(trace_id="trace-1")
    root.inputs = {"messages": [{"type": "human", "content": "What is 2+2?"}]}
    root.outputs = {"messages": [{"type": "ai", "content": "2+2 = 4."}]}

    guardrails = make_run(
        "guardrails", run_type=RunType.CHAIN, name="guardrails", parent="root", trace_id="trace-1"
    )
    guardrails.outputs = {"guardrail": "safe"}

    middleware = make_run(
        "mw",
        run_type=RunType.CHAIN,
        name="ExampleMiddleware.awrap_model_call",
        parent="root",
        trace_id="trace-1",
    )

    llm1 = make_run(
        "llm-1",
        run_type=RunType.LLM,
        name="ChatOpenAI",
        parent="root",
        trace_id="trace-1",
        total_tokens=50,
        prompt_tokens=40,
    )
    llm1.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "What is 2+2? Use the calculator tool."),
    ]
    llm1.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="calculator", args={"expression": "2+2"}, id="call-1")],
    )

    tool1 = make_run(
        "tool-1", run_type=RunType.TOOL, name="calculator", parent="root", trace_id="trace-1"
    )
    tool1.inputs = {"expression": "2+2"}
    tool1.outputs = {"output": {"update": {"messages": [{"content": "4", "type": "tool"}]}}}

    llm2 = make_run(
        "llm-2",
        run_type=RunType.LLM,
        name="ChatOpenAI",
        parent="root",
        trace_id="trace-1",
        total_tokens=50,
        prompt_tokens=45,
    )
    llm2.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "What is 2+2? Use the calculator tool."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="calculator", args={"expression": "2+2"}, id="call-1")],
        ),
        Message(role="tool", text="4", tool_call_id="call-1"),
    ]
    llm2.output_message = Message(role="ai", text="2+2 = 4.")

    return [root, guardrails, middleware, llm1, tool1, llm2]


def test_significant_runs_filters_middleware():
    runs = _mini_runs()
    kept = significant_runs(runs)
    names = [r.name for r in kept]
    assert "ExampleMiddleware.awrap_model_call" not in names


def test_significant_runs_keeps_root_llm_tool_and_first_level():
    runs = _mini_runs()
    kept = significant_runs(runs)
    ids = {r.id for r in kept}
    assert {"root", "run-guardrails", "run-llm-1", "run-tool-1", "run-llm-2"} == ids


def test_skeleton_contains_one_line_per_significant_run():
    runs = _mini_runs()
    skeleton = build_skeleton(runs)
    assert "trace-1" in skeleton
    assert skeleton.count("[llm]") == 2
    assert skeleton.count("[tool]") == 1
    assert "Middleware" not in skeleton
    assert "id=run-llm-1" in skeleton


def test_narrative_first_llm_call_shows_all_messages():
    runs = _mini_runs()
    narrative = build_narrative(runs)
    assert "You are a helpful agent." in narrative
    assert "What is 2+2?" in narrative


def test_narrative_second_llm_call_is_delta():
    runs = _mini_runs()
    narrative = build_narrative(runs)
    assert narrative.count("You are a helpful agent.") == 1
    assert "unchanged messages" in narrative


def test_narrative_includes_tool_call_and_result():
    runs = _mini_runs()
    narrative = build_narrative(runs)
    assert "calculator" in narrative
    assert "expression" in narrative
    assert "result: 4" in narrative


def test_narrative_includes_first_level_nodes():
    runs = _mini_runs()
    narrative = build_narrative(runs)
    assert "node guardrails" in narrative
    assert "safe" in narrative


def test_narrative_includes_final_response():
    runs = _mini_runs()
    narrative = build_narrative(runs)
    assert "2+2 = 4." in narrative


def test_run_detail_full_context():
    runs = _mini_runs()
    detail = run_detail(runs, "run-llm-2")
    assert detail.count("You are a helpful agent.") == 1
    assert "2+2 = 4." in detail


def test_run_detail_unknown_id_raises():
    runs = _mini_runs()
    import pytest

    with pytest.raises(ValueError):
        run_detail(runs, "nope")


def test_empty_trace_does_not_crash():
    assert "(empty trace)" in build_skeleton([])
    assert "(empty trace)" in build_narrative([])


def test_skeleton_labels_infra_cancelled_error():
    """A CancelledError is labeled as infra-cancelled, not an agent error."""
    root = make_root(trace_id="trace-cancel")
    root.error = "CancelledError()"
    root.status = "error"
    skeleton = build_skeleton([root])
    assert "infra-cancelled" in skeleton
    assert "not an agent failure" in skeleton


def test_skeleton_labels_regular_error_normally():
    """A regular error is labeled as ERROR without the infra-cancelled tag."""
    root = make_root(trace_id="trace-err")
    root.error = "ValueError: bad input"
    root.status = "error"
    skeleton = build_skeleton([root])
    assert "ERROR" in skeleton
    assert "infra-cancelled" not in skeleton


def test_narrative_labels_infra_cancelled_error():
    """A CancelledError in a tool run is labeled in the narrative too."""
    root = make_root(trace_id="trace-cancel")
    root.inputs = {"messages": [{"type": "human", "content": "do something"}]}
    root.outputs = {"messages": [{"type": "ai", "content": "ok"}]}
    tool = make_run(
        "tool-1", run_type=RunType.TOOL, name="slow_tool", parent="root", trace_id="trace-cancel"
    )
    tool.error = "asyncio.exceptions.CancelledError"
    narrative = build_narrative([root, tool])
    assert "infra-cancelled" in narrative


def test_skeleton_labels_timeout_as_infra():
    """TimeoutError is also recognized as infra-cancelled."""
    root = make_root(trace_id="trace-timeout")
    root.error = "TimeoutError: operation timed out"
    root.status = "error"
    skeleton = build_skeleton([root])
    assert "infra-cancelled" in skeleton


# ---------------------------------------------------------------------------
# Narrative compact mode (per-index diff)
# ---------------------------------------------------------------------------


def _dynamic_prompt_runs():
    """Build runs where the system message changes every step (dynamic prompt).

    This is the pattern that breaks the common-prefix walk: the system message
    changes by one character each step, so common=0 and the entire message list
    is re-dumped every step in full mode.

    Uses a long human message so the compact mode savings are measurable.
    """
    long_task = (
        "Write a song about a sailor who returns home after 20 years at sea. "
        "The song should have three verses and a chorus. Make it melancholic "
        "but with a hopeful ending. Use nautical imagery throughout."
    )
    root = make_root(trace_id="trace-dyn")
    root.inputs = {"messages": [{"type": "human", "content": long_task}]}
    root.outputs = {"messages": [{"type": "ai", "content": "ok"}]}

    llm1 = make_run(
        "llm-1", run_type=RunType.LLM, name="ChatOpenAI", parent="root", trace_id="trace-dyn"
    )
    llm1.input_messages = [
        make_msg("system", "You are a songwriter. Draft: v1"),
        make_msg("human", long_task),
    ]
    llm1.output_message = Message(role="ai", text="Verse 1...")

    tool1 = make_run(
        "tool-1", run_type=RunType.TOOL, name="edit_file", parent="root", trace_id="trace-dyn"
    )
    tool1.inputs = {"path": "song.md", "content": "Verse 1..."}
    tool1.outputs = {"output": "ok"}

    llm2 = make_run(
        "llm-2", run_type=RunType.LLM, name="ChatOpenAI", parent="root", trace_id="trace-dyn"
    )
    llm2.input_messages = [
        make_msg("system", "You are a songwriter. Draft: v2"),
        make_msg("human", long_task),
        Message(role="ai", text="Verse 1..."),
        Message(role="tool", text="ok", tool_call_id="call-1"),
    ]
    llm2.output_message = Message(role="ai", text="Verse 2...")

    return [root, llm1, tool1, llm2]


def test_narrative_compact_collapses_unchanged_messages():
    """Compact mode collapses unchanged messages even when the system prompt changed."""
    runs = _dynamic_prompt_runs()
    narrative = build_narrative(runs, mode="compact")
    # The long human message appears in Task section and Step 1 (first call shows all).
    # It should NOT be re-dumped in Step 2 (collapsed by per-index diff).
    long_task = "Write a song about a sailor"
    assert narrative.count(long_task) == 2  # Task + Step 1, but NOT Step 2


def test_narrative_full_redumps_after_first_change():
    """Full mode re-dumps all messages after the first change (common-prefix walk)."""
    runs = _dynamic_prompt_runs()
    narrative = build_narrative(runs, mode="full")
    # The long human message appears in Task + Step 1 + Step 2 (re-dumped because system changed)
    long_task = "Write a song about a sailor"
    assert narrative.count(long_task) >= 3


def test_narrative_compact_smaller_than_full():
    """Compact mode produces a smaller file than full mode on dynamic prompts."""
    runs = _dynamic_prompt_runs()
    compact = build_narrative(runs, mode="compact")
    full = build_narrative(runs, mode="full")
    assert len(compact) < len(full)


def test_narrative_compact_shows_unchanged_marker():
    """Compact mode shows '(N unchanged messages)' for collapsed messages."""
    runs = _dynamic_prompt_runs()
    narrative = build_narrative(runs, mode="compact")
    assert "unchanged messages" in narrative


def test_narrative_default_mode_is_compact():
    """build_narrative defaults to compact mode."""
    runs = _dynamic_prompt_runs()
    default = build_narrative(runs)
    compact = build_narrative(runs, mode="compact")
    assert default == compact


def test_narrative_compact_still_shows_changed_system():
    """Compact mode still shows the system message when it changes."""
    runs = _dynamic_prompt_runs()
    narrative = build_narrative(runs, mode="compact")
    assert "Draft: v1" in narrative
    assert "Draft: v2" in narrative
