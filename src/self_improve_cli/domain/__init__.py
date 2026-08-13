"""Canonical trace model.

All layers (privacy, storage, representations, metrics, CLI) operate on these
types — never on provider-specific SDK objects. Source adapters normalize
provider payloads into this model at the boundary.

The model is deliberately narrow: it captures what agent traces commonly
expose (runs, messages, tool calls, token counts, errors) without trying to
represent every provider-specific field. Unknown fields are preserved in
`extra` for round-tripping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Schema version for canonical artifacts. Bump when the on-disk format changes
# in a backwards-incompatible way.
SCHEMA_VERSION = 1


class RunType(StrEnum):
    """Type of a run within a trace."""

    LLM = "llm"
    TOOL = "tool"
    CHAIN = "chain"
    OTHER = "other"


@dataclass
class ToolCall:
    """A tool call requested by an LLM response."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class Message:
    """A normalized message within an LLM run's input or output.

    role: one of "system", "human", "ai", "tool", or a fallback string.
    text: the plain-text content of the message.
    tool_calls: tool calls attached to the message (typically AI messages).
    tool_call_id: for tool messages, the ID of the call that produced this result.
    """

    role: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class Run:
    """A single unit of work within a trace.

    Canonical runs are provider-agnostic. Source adapters are responsible for
    normalizing provider payloads into this shape, including message parsing,
    token extraction, and error normalization.
    """

    id: str
    trace_id: str
    run_type: RunType
    name: str
    parent_run_id: str | None = None
    dotted_order: str | None = None
    status: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    input_messages: list[Message] = field(default_factory=list)
    output_message: Message | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """A complete trace: a collection of runs for one end-to-end operation."""

    trace_id: str
    runs: list[Run]
    schema_version: int = SCHEMA_VERSION
    sanitized: bool = False
    sanitization_report: dict[str, Any] | None = None
    source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    """Lightweight run listing for L0 discovery (no full content)."""

    id: str
    trace_id: str
    name: str
    run_type: RunType
    status: str | None
    start_time: str | None
    total_tokens: int | None
    error: str | None = None


def run_to_summary(run: Run) -> RunSummary:
    """Project a full Run into a lightweight RunSummary."""
    return RunSummary(
        id=run.id,
        trace_id=run.trace_id,
        name=run.name,
        run_type=run.run_type,
        status=run.status,
        start_time=run.start_time,
        total_tokens=run.total_tokens,
        error=run.error,
    )


@dataclass
class RunResolution:
    """Result of resolving a run ID to its trace and project.

    trace_id: the parent trace ID of the run.
    project_id: the LangSmith session/project UUID the run belongs to.
        Use this as ``project_id`` in ``list_runs`` to fetch the full trace
        from the correct project, even when it differs from the configured
        default project.
    """

    trace_id: str
    project_id: str | None = None


@dataclass
class ProjectSummary:
    """Lightweight project listing for L0 discovery."""

    id: str
    name: str
    run_count: int | None = None
