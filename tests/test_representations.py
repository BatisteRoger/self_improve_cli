"""Tests for progressive representations (TER)."""

from uuid import uuid4

from self_improve_cli.domain import Message, Run, RunType, ToolCall
from self_improve_cli.representations import (
    build_narrative,
    build_skeleton,
    context_at,
    context_at_data,
    error_neighborhood,
    error_neighborhood_data,
    narrative_data,
    run_detail,
    run_detail_data,
    significant_runs,
    target_timeline,
    target_timeline_data,
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

    with pytest.raises(ValueError, match="not found in trace"):
        run_detail(runs, "nope")


def test_run_detail_unknown_id_suggests_skeleton():
    """Recoverable error: the message should suggest how to find valid run IDs."""
    runs = _mini_runs()
    import pytest

    with pytest.raises(ValueError, match="self-improve skeleton"):
        run_detail(runs, "nope")


def test_run_detail_shows_parent_and_child_refs():
    """Connectedness: run-detail should expose parent and child run references."""
    runs = _mini_runs()
    # run-llm-2 is a child of the root run; check that parent is shown.
    detail = run_detail(runs, "run-llm-2")
    assert "## Related runs" in detail
    assert "Parent:" in detail


def test_run_detail_no_related_runs_section_when_isolated():
    """A root run with no children should not show an empty Related runs section."""
    from tests.helpers import make_root

    single = [make_root()]
    detail = run_detail(single, single[0].id)
    assert "## Related runs" not in detail


def test_run_detail_tool_calls_only_shows_tool_calls():
    """--tool-calls-only shows tool calls without prompts or output text."""
    runs = _mini_runs()
    detail = run_detail(runs, "run-llm-1", tool_calls_only=True)
    assert "calculator" in detail
    assert "expression" in detail
    assert "## Input messages" not in detail
    assert "## Output" not in detail
    assert "## Related runs" not in detail


def test_run_detail_tool_calls_only_no_calls():
    """--tool-calls-only on a run with no tool calls says so honestly."""
    runs = _mini_runs()
    detail = run_detail(runs, "run-llm-2", tool_calls_only=True)
    assert "no tool calls" in detail


def test_run_detail_tool_calls_only_non_llm():
    """--tool-calls-only on a non-LLM run reports it's for LLM runs."""
    runs = _mini_runs()
    detail = run_detail(runs, "run-tool-1", tool_calls_only=True)
    assert "LLM runs" in detail


def test_run_detail_data_tool_calls_only():
    """run_detail_data with tool_calls_only returns structured tool calls."""
    runs = _mini_runs()
    data = run_detail_data(runs, "run-llm-1", tool_calls_only=True)
    assert "tool_calls" in data
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["name"] == "calculator"
    assert "input_messages" not in data


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


# ---------------------------------------------------------------------------
# SLN-36: narrative step-range filter
# ---------------------------------------------------------------------------


def test_narrative_step_from_filters_early_steps():
    """--from N excludes steps before N but keeps the header and task."""
    runs = _mini_runs()
    narrative = build_narrative(runs, step_from=2)
    assert "# Narrative" in narrative
    assert "## Task" in narrative
    assert "showing steps 2-" in narrative
    assert "## Step 1 —" not in narrative
    assert "## Step 2 —" in narrative


def test_narrative_step_to_filters_late_steps():
    """--to N excludes steps after N."""
    runs = _mini_runs()
    narrative = build_narrative(runs, step_to=1)
    assert "## Step 1 —" in narrative
    assert "## Step 2 —" not in narrative
    assert "## Step 3 —" not in narrative


def test_narrative_step_range_both():
    """--from N --to M shows only steps in [N, M]."""
    runs = _mini_runs()
    narrative = build_narrative(runs, step_from=2, step_to=3)
    assert "## Step 1 —" not in narrative
    assert "## Step 2 —" in narrative
    assert "## Step 3 —" in narrative
    assert "## Step 4 —" not in narrative


def test_narrative_step_range_includes_total():
    """Range note shows total steps."""
    runs = _mini_runs()
    narrative = build_narrative(runs, step_from=2)
    assert "of " in narrative


def test_narrative_no_range_shows_all():
    """Without range filter, all steps are shown and no range note appears."""
    runs = _mini_runs()
    narrative = build_narrative(runs)
    assert "showing steps" not in narrative
    assert "## Step 1 —" in narrative
    assert "## Step 4 —" in narrative


def test_narrative_data_step_range():
    """narrative_data respects step_from/step_to and includes range metadata."""
    runs = _mini_runs()
    data = narrative_data(runs, step_from=2, step_to=3)
    steps = data["steps"]
    assert all(s["step"] >= 2 for s in steps)
    assert all(s["step"] <= 3 for s in steps)
    assert data["total_steps"] >= 3
    assert data["step_range"] == {"from": 2, "to": 3, "total": data["total_steps"]}


def test_narrative_data_no_range():
    """narrative_data without range has no step_range key."""
    runs = _mini_runs()
    data = narrative_data(runs)
    assert "step_range" not in data
    assert "total_steps" in data


# ---------------------------------------------------------------------------
# SLN-33: significant_runs type consistency (UUID vs str parent_run_id)
# ---------------------------------------------------------------------------


def test_significant_runs_uuid_parent_run_id_matches_str_root_id():
    """SLN-33: a UUID parent_run_id must match a str root id.

    The LangSmith SDK can return parent_run_id as a UUID object while id is
    already str-cast.  Before the fix, ``UUID in {str}`` was always False,
    so first-level chain runs were silently dropped during fetch but kept
    after save/load (json.dumps coerces UUIDs to strings).  This test
    reproduces that scenario directly.
    """
    root_uid = uuid4()
    root = Run(
        id=str(root_uid),
        trace_id="trace-uuid",
        run_type=RunType.CHAIN,
        name="agent",
        parent_run_id=None,
        dotted_order=f"0001{root_uid}",
        status="success",
    )
    # child with UUID parent_run_id (as the SDK returns it before the fix)
    child = Run(
        id="child-1",
        trace_id="trace-uuid",
        run_type=RunType.CHAIN,
        name="subgraph",
        parent_run_id=root_uid,  # type: ignore[arg-type]  # UUID, not str (reproduces SDK bug)
        dotted_order=f"0001{root_uid}.0002child-1",
        status="success",
    )
    kept = significant_runs([root, child])
    ids = {r.id for r in kept}
    assert "child-1" in ids, "first-level chain run with UUID parent_run_id was dropped"


def test_significant_runs_consistent_across_types():
    """SLN-33: significant_runs must give the same result whether parent_run_id
    is a UUID or its string form."""
    root_uid = uuid4()
    root = Run(
        id=str(root_uid),
        trace_id="trace-consistency",
        run_type=RunType.CHAIN,
        name="agent",
        parent_run_id=None,
        dotted_order=f"0001{root_uid}",
        status="success",
    )
    child_uuid = Run(
        id="child-uuid",
        trace_id="trace-consistency",
        run_type=RunType.CHAIN,
        name="subgraph",
        parent_run_id=root_uid,  # type: ignore[arg-type]  # UUID, not str (reproduces SDK bug)
        dotted_order=f"0001{root_uid}.0002child-uuid",
        status="success",
    )
    child_str = Run(
        id="child-str",
        trace_id="trace-consistency",
        run_type=RunType.CHAIN,
        name="subgraph",
        parent_run_id=str(root_uid),
        dotted_order=f"0001{root_uid}.0003child-str",
        status="success",
    )
    kept_uuid = significant_runs([root, child_uuid])
    kept_str = significant_runs([root, child_str])
    assert len(kept_uuid) == len(kept_str), (
        f"UUID vs str parent_run_id gave different counts: {len(kept_uuid)} vs {len(kept_str)}"
    )


# ---------------------------------------------------------------------------
# SLN-6: context-at
# ---------------------------------------------------------------------------


def test_context_at_step_zero_shows_first_input_messages():
    """context-at step 0 shows the first main-loop LLM run's input messages."""
    runs = _mini_runs()
    out = context_at(runs, 0)
    assert "Context-at step 0" in out
    assert "You are a helpful agent." in out
    assert "What is 2+2?" in out


def test_context_at_step_one_shows_tool_result():
    """context-at step 1 shows the tool result that was added between steps."""
    runs = _mini_runs()
    out = context_at(runs, 1)
    assert "Context-at step 1" in out
    # The tool result message should be visible
    assert "4" in out


def test_context_at_out_of_range_raises_with_recovery():
    """An out-of-range step raises a recoverable error listing valid steps."""
    runs = _mini_runs()
    import pytest

    with pytest.raises(ValueError, match="out of range"):
        context_at(runs, 99)
    with pytest.raises(ValueError, match="out of range"):
        context_at(runs, -1)


def test_context_at_inputs_only_omits_output():
    """--inputs-only shows input messages but not the output section."""
    runs = _mini_runs()
    out = context_at(runs, 0, inputs_only=True)
    assert "## Input messages" in out
    assert "## Output" not in out


def test_context_at_outputs_only_omits_inputs():
    """--outputs-only shows the output but not the input messages section."""
    runs = _mini_runs()
    out = context_at(runs, 0, outputs_only=True)
    assert "## Output" in out
    assert "## Input messages" not in out


def test_context_at_outputs_only_shows_tool_calls():
    """--outputs-only shows tool calls in the output."""
    runs = _mini_runs()
    out = context_at(runs, 0, outputs_only=True)
    assert "calculator" in out


def test_context_at_data_outputs_only():
    """context_at_data with outputs_only omits messages key."""
    runs = _mini_runs()
    data = context_at_data(runs, 0, outputs_only=True)
    assert "messages" not in data
    assert "output" in data


def test_context_at_full_does_not_truncate():
    """--full shows the full message text without truncation markers."""
    long_text = "x" * 1000
    root = make_root(trace_id="trace-long")
    llm = make_run(
        "llm-1", run_type=RunType.LLM, name="ChatOpenAI", parent="root", trace_id="trace-long"
    )
    llm.input_messages = [make_msg("human", long_text)]
    llm.output_message = Message(role="ai", text="ok")
    out_bounded = context_at([root, llm], 0)
    out_full = context_at([root, llm], 0, full=True)
    assert "truncated" in out_bounded
    assert "truncated" not in out_full
    assert long_text in out_full


def test_context_at_tool_filter_isolates_one_result():
    """--tool <tool_call_id> shows only the matching AI tool_call and tool result."""
    runs = _mini_runs()
    out = context_at(runs, 1, tool_call_id="call-1")
    assert "tool_call_id=call-1" in out
    # The system message should NOT appear when filtered to one tool call
    assert "You are a helpful agent." not in out


def test_context_at_tool_filter_missing_id_reports_honestly():
    """A non-existent tool_call_id is reported, not silently empty."""
    runs = _mini_runs()
    out = context_at(runs, 1, tool_call_id="no-such-call")
    assert "No message with tool_call_id=no-such-call" in out


def test_context_at_diff_shows_added_messages():
    """--from 0 --to 1 shows the tool result as added/changed."""
    runs = _mini_runs()
    out = context_at(runs, 0, from_step=0, to_step=1)
    assert "diff" in out.lower()
    # The tool result message (role=tool, text=4) is new at step 1
    assert "tool" in out.lower()


def test_context_at_diff_identical_steps_says_no_differences():
    """Diffing the same step against itself reports no differences."""
    runs = _mini_runs()
    out = context_at(runs, 0, from_step=0, to_step=0)
    assert "No differences" in out


def test_context_at_diff_shrink_notes_inferred_compaction():
    """When the message list shrinks, the diff notes inferred compaction honestly."""
    root = make_root(trace_id="trace-shrink")
    # Step 0: 5 messages
    llm1 = make_run("llm-1", run_type=RunType.LLM, name="M", parent="root", trace_id="trace-shrink")
    llm1.input_messages = [make_msg("system", f"msg {i}") for i in range(5)]
    llm1.output_message = Message(role="ai", text="ok")
    # Step 1: 2 messages (apparent compaction)
    llm2 = make_run("llm-2", run_type=RunType.LLM, name="M", parent="root", trace_id="trace-shrink")
    llm2.input_messages = [make_msg("system", "compacted"), make_msg("human", "continue")]
    llm2.output_message = Message(role="ai", text="ok")
    out = context_at([root, llm1, llm2], 0, from_step=0, to_step=1)
    assert "inferred" in out.lower()


def test_context_at_navigation_links_to_prev_and_next():
    """context-at shows navigation references to previous and next steps."""
    runs = _mini_runs()
    out = context_at(runs, 1)
    assert "Previous step 0" in out
    # Step 1 is the last step, so no next link
    assert "Next step" not in out


def test_context_at_data_is_structured():
    """context_at_data returns structured dict, not markdown."""
    runs = _mini_runs()
    data = context_at_data(runs, 0)
    assert isinstance(data, dict)
    assert data["step"] == 0
    assert data["run_id"] == "run-llm-1"
    assert isinstance(data["messages"], list)
    assert data["messages"][0]["role"] == "system"
    assert "navigation" in data


def test_context_at_data_diff_is_structured():
    """context_at_data with from/to returns structured diff."""
    runs = _mini_runs()
    data = context_at_data(runs, 0, from_step=0, to_step=1)
    assert data["from_step"] == 0
    assert data["to_step"] == 1
    assert isinstance(data.get("changed"), list) or isinstance(data.get("added"), list)


def test_context_at_empty_trace_handles_gracefully():
    """An empty trace does not crash context-at."""
    out = context_at([], 0)
    assert "no main-loop LLM runs" in out


# ---------------------------------------------------------------------------
# SLN-12: target-timeline + error-neighborhood
# ---------------------------------------------------------------------------


def test_target_timeline_finds_matching_runs():
    """target-timeline finds runs whose inputs/outputs mention the target."""
    runs = _mini_runs()
    out = target_timeline(runs, "calculator")
    assert "Target timeline" in out
    # The calculator tool run should be in the timeline
    assert "calculator" in out
    assert "run-tool-1" in out


def test_target_timeline_no_matches_reports_honestly():
    """target-timeline reports when no run touched the target."""
    runs = _mini_runs()
    out = target_timeline(runs, "nonexistent-target")
    assert "No significant run touched" in out


def test_target_timeline_includes_run_ids_for_drill_down():
    """target-timeline includes run-detail navigation references."""
    runs = _mini_runs()
    out = target_timeline(runs, "calculator")
    assert "run-detail" in out


def test_target_timeline_data_is_structured():
    """target_timeline_data returns structured dict."""
    runs = _mini_runs()
    data = target_timeline_data(runs, "calculator")
    assert data["target"] == "calculator"
    assert isinstance(data["touches"], list)
    assert len(data["touches"]) > 0
    assert "run_id" in data["touches"][0]


def test_error_neighborhood_finds_errors():
    """error-neighborhood finds runs with errors and shows the neighborhood."""
    root = make_root(trace_id="trace-err")
    root.inputs = {"messages": [{"type": "human", "content": "do something"}]}
    root.outputs = {"messages": [{"type": "ai", "content": "ok"}]}
    tool_err = make_run(
        "tool-1", run_type=RunType.TOOL, name="failing_tool", parent="root", trace_id="trace-err"
    )
    tool_err.inputs = {"path": "file.txt"}
    tool_err.error = "ValueError: bad input"
    tool_err.status = "error"
    llm_after = make_run(
        "llm-1", run_type=RunType.LLM, name="ChatOpenAI", parent="root", trace_id="trace-err"
    )
    llm_after.input_messages = [make_msg("system", "sys"), make_msg("human", "retry")]
    llm_after.output_message = Message(role="ai", text="trying again")
    out = error_neighborhood([root, tool_err, llm_after])
    assert "Error neighborhood" in out
    assert "failing_tool" in out
    assert "ValueError" in out
    # Should show the agent's reaction (next step)
    assert "Agent reaction" in out
    assert "ChatOpenAI" in out


def test_error_neighborhood_no_errors_reports_cleanly():
    """error-neighborhood reports when there are no agent errors."""
    runs = _mini_runs()
    out = error_neighborhood(runs)
    assert "No agent errors" in out


def test_error_neighborhood_excludes_infra_cancelled():
    """error-neighborhood excludes infra-cancelled errors (not agent failures)."""
    root = make_root(trace_id="trace-cancel")
    root.error = "CancelledError()"
    root.status = "error"
    out = error_neighborhood([root])
    assert "No agent errors" in out


def test_error_neighborhood_data_is_structured():
    """error_neighborhood_data returns structured dict."""
    root = make_root(trace_id="trace-err2")
    tool_err = make_run(
        "tool-1", run_type=RunType.TOOL, name="bad_tool", parent="root", trace_id="trace-err2"
    )
    tool_err.error = "RuntimeError: oops"
    tool_err.status = "error"
    llm_before = make_run(
        "aaa-1", run_type=RunType.LLM, name="Chat", parent="root", trace_id="trace-err2"
    )
    llm_after = make_run(
        "zzz-1", run_type=RunType.LLM, name="ChatAfter", parent="root", trace_id="trace-err2"
    )
    data = error_neighborhood_data([root, tool_err, llm_before, llm_after])
    assert data["error_count"] == 1
    assert len(data["errors"]) == 1
    err = data["errors"][0]
    assert err["error"] == "RuntimeError: oops"
    assert isinstance(err["neighborhood"], list)
    assert err["agent_reaction"] is not None


def test_error_neighborhood_empty_trace():
    """error-neighborhood handles an empty trace gracefully."""
    out = error_neighborhood([])
    assert "empty trace" in out
