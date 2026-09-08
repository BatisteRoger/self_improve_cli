"""Benchmark spec: metadata for each synthetic diagnostic trace.

Each entry describes:
- id: a stable identifier for the trace
- description: what the trace contains
- known_issue: the issue an analyst should find (or "none" for clean controls)
- should_not_find: things the analyst should NOT flag (false-positive guards)
- evidence_run_ids: run IDs that contain the primary evidence
- expected_next_check: what a good analyst should suggest verifying next

The spec is consumed by the test harness and by any future CLI benchmark
command. It is also documentation: a new contributor can read this file
to understand what each benchmark trace tests.
"""

from __future__ import annotations

from typing import Any

BENCHMARK_TRACES: list[dict[str, Any]] = [
    {
        "id": "ignored_tool_error",
        "description": (
            "Agent continued after a tool returned a 403 error without adapting its approach."
        ),
        "known_issue": "ignored_tool_error",
        "should_not_find": ["redundant_work", "unverified_completion"],
        "evidence_run_ids": ["run-tool-1", "run-llm-2"],
        "expected_next_check": (
            "Check whether the 403 error appeared in the subsequent "
            "model input (context-at or run-detail)."
        ),
    },
    {
        "id": "unverified_completion",
        "description": (
            "Agent claimed success without running any verification after the final edit."
        ),
        "known_issue": "unverified_completion",
        "should_not_find": ["ignored_tool_error", "redundant_work"],
        "evidence_run_ids": ["run-llm-3"],
        "expected_next_check": (
            "Check whether any tool call after the edit verifies the "
            "result (e.g. read_file, test run)."
        ),
    },
    {
        "id": "legitimate_repetition",
        "description": (
            "Agent read a file multiple times because it changed "
            "between reads. Should NOT be flagged as redundant."
        ),
        "known_issue": "none",
        "should_not_find": [
            "redundant_work",
            "ignored_tool_error",
            "unverified_completion",
        ],
        "evidence_run_ids": ["run-tool-1", "run-tool-2"],
        "expected_next_check": (
            "Verify that the file content changed between reads (diff the tool outputs)."
        ),
    },
    {
        "id": "lost_constraint_after_compaction",
        "description": (
            "A user constraint disappeared from the context and the agent violated it."
        ),
        "known_issue": "lost_constraint",
        "should_not_find": ["redundant_work", "ignored_tool_error"],
        "evidence_run_ids": ["run-llm-1", "run-llm-3"],
        "expected_next_check": (
            "Compare the model input at step 1 (constraint present) "
            "with step 3 (constraint absent). Use context-at."
        ),
    },
    {
        "id": "clean_execution",
        "description": (
            "A successful trace that should NOT be criticized. False-positive control."
        ),
        "known_issue": "none",
        "should_not_find": [
            "ignored_tool_error",
            "unverified_completion",
            "redundant_work",
            "lost_constraint",
        ],
        "evidence_run_ids": [],
        "expected_next_check": (
            "None — the trace is clean. If a finding is produced, it is a false positive."
        ),
    },
]


def get_spec(trace_id: str) -> dict[str, Any]:
    """Look up a benchmark trace spec by ID."""
    for spec in BENCHMARK_TRACES:
        if spec["id"] == trace_id:
            return spec
    raise KeyError(f"Unknown benchmark trace: {trace_id}")
