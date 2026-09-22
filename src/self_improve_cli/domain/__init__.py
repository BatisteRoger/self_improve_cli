"""Canonical trace model and analyst records.

All layers (privacy, storage, representations, metrics, CLI) operate on these
types — never on provider-specific SDK objects. Source adapters normalize
provider payloads into this model at the boundary.

- `trace` — runs, messages, tool calls, token counts, errors.
- `records` — analyst-authored metadata (assessments, findings) stored
  alongside traces.
"""

from self_improve_cli.domain.records import (
    Assessment,
    Finding,
    OutcomeSource,
    OutcomeStatus,
)
from self_improve_cli.domain.trace import (
    SCHEMA_VERSION,
    Message,
    ProjectSummary,
    Run,
    RunResolution,
    RunSummary,
    RunType,
    ToolCall,
    Trace,
    run_to_summary,
)

__all__ = [
    "SCHEMA_VERSION",
    "Assessment",
    "Finding",
    "Message",
    "OutcomeSource",
    "OutcomeStatus",
    "ProjectSummary",
    "Run",
    "RunResolution",
    "RunSummary",
    "RunType",
    "ToolCall",
    "Trace",
    "run_to_summary",
]
