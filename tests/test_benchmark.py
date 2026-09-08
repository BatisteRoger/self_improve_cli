"""Tests for the diagnostic benchmark traces.

These tests verify that each synthetic trace is correctly constructed and
contains the evidence it claims to contain. They do NOT test the analyst
agent — that is a separate evaluation step.

The benchmark itself is the infrastructure; the evaluation is run separately
(manually or via a future CLI command).
"""

from __future__ import annotations

import pytest

from self_improve_cli.representations import build_narrative, build_skeleton, run_detail
from tests.fixtures.benchmark.spec import BENCHMARK_TRACES, get_spec
from tests.fixtures.benchmark.traces import BUILDERS, build_trace

# ---------------------------------------------------------------------------
# Spec integrity
# ---------------------------------------------------------------------------


def test_all_specs_have_builders():
    """Every spec entry must have a corresponding builder."""
    for spec in BENCHMARK_TRACES:
        assert spec["id"] in BUILDERS, f"No builder for {spec['id']}"


def test_all_builders_have_specs():
    """Every builder must have a corresponding spec entry."""
    spec_ids = {s["id"] for s in BENCHMARK_TRACES}
    for builder_id in BUILDERS:
        assert builder_id in spec_ids, f"No spec for {builder_id}"


def test_spec_fields_are_complete():
    """Every spec must have all required fields."""
    required = {"id", "description", "known_issue", "should_not_find", "evidence_run_ids"}
    for spec in BENCHMARK_TRACES:
        missing = required - set(spec.keys())
        assert not missing, f"Spec {spec['id']} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Trace construction integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace_id", list(BUILDERS.keys()))
def test_trace_has_runs(trace_id):
    """Every benchmark trace must have at least one run."""
    runs = build_trace(trace_id)
    assert len(runs) > 0


@pytest.mark.parametrize("trace_id", list(BUILDERS.keys()))
def test_trace_has_root(trace_id):
    """Every benchmark trace must have a root run (no parent)."""
    runs = build_trace(trace_id)
    roots = [r for r in runs if r.parent_run_id is None]
    assert len(roots) == 1, f"{trace_id}: expected 1 root, found {len(roots)}"


@pytest.mark.parametrize("trace_id", list(BUILDERS.keys()))
def test_trace_ids_consistent(trace_id):
    """All runs in a trace must share the same trace_id."""
    runs = build_trace(trace_id)
    for r in runs:
        assert r.trace_id == trace_id, f"{trace_id}: run {r.id} has trace_id={r.trace_id}"


@pytest.mark.parametrize("trace_id", list(BUILDERS.keys()))
def test_evidence_run_ids_exist(trace_id):
    """Every evidence_run_id in the spec must exist in the trace."""
    spec = get_spec(trace_id)
    runs = build_trace(trace_id)
    run_ids = {r.id for r in runs}
    for rid in spec["evidence_run_ids"]:
        assert rid in run_ids, f"{trace_id}: evidence run {rid} not found in trace"


@pytest.mark.parametrize("trace_id", list(BUILDERS.keys()))
def test_trace_produces_representations(trace_id):
    """Every benchmark trace must produce valid skeleton, narrative, and run-detail."""
    runs = build_trace(trace_id)
    skeleton = build_skeleton(runs)
    assert "# Skeleton" in skeleton
    narrative = build_narrative(runs)
    assert "# Narrative" in narrative
    # run-detail on the first LLM run should work
    llm_runs = [r for r in runs if r.run_type.value == "llm"]
    if llm_runs:
        detail = run_detail(runs, llm_runs[0].id)
        assert "# Run detail" in detail


# ---------------------------------------------------------------------------
# Issue-specific verification
# ---------------------------------------------------------------------------


def test_ignored_tool_error_has_error_in_tool_output():
    """The ignored_tool_error trace must have a tool error in the tool output."""
    runs = build_trace("ignored_tool_error")
    tool_runs = [r for r in runs if r.run_type.value == "tool"]
    assert any(r.error for r in tool_runs), "Expected at least one tool with an error"


def test_ignored_tool_error_agent_ignores_error():
    """The agent's final response must not acknowledge the error."""
    runs = build_trace("ignored_tool_error")
    llm_runs = [r for r in runs if r.run_type.value == "llm"]
    final = llm_runs[-1]
    assert final.output_message is not None
    # The agent says "Done!" despite the error — that's the issue.
    assert "Done" in (final.output_message.text or "")


def test_unverified_completion_has_no_verification_after_edit():
    """The unverified_completion trace must have no verification tool call after the edit."""
    runs = build_trace("unverified_completion")
    # Find the last edit_file tool call
    last_edit_idx = None
    for i, r in enumerate(runs):
        if r.run_type.value == "tool" and r.name == "edit_file":
            last_edit_idx = i
    assert last_edit_idx is not None, "Expected at least one edit_file call"
    # No tool call after the last edit should be a verification (read_file, test, etc.)
    for r in runs[last_edit_idx + 1 :]:
        if r.run_type.value == "tool":
            assert r.name not in ("read_file", "test", "run_tests", "diff"), (
                f"Unexpected verification call after edit: {r.name}"
            )


def test_legitimate_repetition_file_changed_between_reads():
    """The legitimate_repetition trace must show different content on each read."""
    runs = build_trace("legitimate_repetition")
    read_results = [
        r.outputs.get("content", "")
        for r in runs
        if r.run_type.value == "tool" and r.name == "read_file"
    ]
    assert len(read_results) >= 2, "Expected at least 2 read_file calls"
    assert read_results[0] != read_results[-1], "File content should differ between reads"


def test_lost_constraint_constraint_absent_in_later_step():
    """The lost_constraint trace must show the constraint absent in a later LLM step."""
    runs = build_trace("lost_constraint_after_compaction")
    llm_runs = [r for r in runs if r.run_type.value == "llm"]
    assert len(llm_runs) >= 2
    first = llm_runs[0]
    later = llm_runs[1]
    # The first step should contain the constraint
    first_texts = " ".join(m.text for m in first.input_messages)
    assert "without using the network" in first_texts
    # The later step should NOT contain the constraint
    later_texts = " ".join(m.text for m in later.input_messages)
    assert "without using the network" not in later_texts


def test_clean_execution_has_no_errors():
    """The clean_execution trace must have no errors in any run."""
    runs = build_trace("clean_execution")
    for r in runs:
        assert r.error is None, f"Clean trace should have no errors, but {r.id} has: {r.error}"


def test_clean_execution_verifies_result():
    """The clean_execution trace must verify the result after the edit."""
    runs = build_trace("clean_execution")
    # Find the last edit_file tool call
    last_edit_idx = None
    for i, r in enumerate(runs):
        if r.run_type.value == "tool" and r.name == "edit_file":
            last_edit_idx = i
    assert last_edit_idx is not None
    # There should be a read_file after the edit
    after_edit = runs[last_edit_idx + 1 :]
    assert any(r.name == "read_file" for r in after_edit if r.run_type.value == "tool"), (
        "Expected a read_file verification after the edit"
    )
