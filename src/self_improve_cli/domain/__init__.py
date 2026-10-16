"""Canonical trace model and analyst records.

All layers (privacy, storage, representations, metrics, CLI) operate on these
types — never on provider-specific SDK objects. Source adapters normalize
provider payloads into this model at the boundary.

- `trace` — runs, messages, tool calls, token counts, errors.
- `records` — analyst-authored metadata (assessments, findings) stored
  alongside traces.
"""

from self_improve_cli.domain.experiment import (
    ExperimentListing,
    ExperimentRunRef,
    OutcomePolicy,
    outcome_from_policy,
    parse_outcome_key,
)
from self_improve_cli.domain.records import (
    Assessment,
    FeedbackScore,
    Finding,
    OutcomeSource,
    OutcomeStatus,
    autofill_assessment,
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
    "ExperimentListing",
    "ExperimentRunRef",
    "OutcomePolicy",
    "FeedbackScore",
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
    "autofill_assessment",
    "outcome_from_policy",
    "parse_outcome_key",
    "run_to_summary",
]
