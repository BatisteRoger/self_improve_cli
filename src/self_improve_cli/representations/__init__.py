"""Token-efficient representations (TER) of traces.

Pure functions over canonical Run objects. No SDK dependency, no LLM calls:
deterministic and recomputable from raw/sanitized data.

Granularity levels:
- L1 skeleton:  one line per significant run.
- L2 narrative: chronological story, message deltas per LLM call.
- L3 run detail: full context of one run, on demand.

One module per view — `common` (shared helpers + significant-run filter),
`skeleton`, `narrative`, `run_detail`, `context_at`, `navigation`
(target-timeline + error-neighborhood), `tools`, and `ter` (orchestration
that writes all derived files at fetch time). All public names are
re-exported here; `from self_improve_cli.representations import X` keeps
working regardless of which module defines X.

Adapted from an internal prototype, reworked around the canonical domain model.
"""

from self_improve_cli.representations.common import significant_runs
from self_improve_cli.representations.context_at import context_at, context_at_data
from self_improve_cli.representations.narrative import build_narrative, narrative_data
from self_improve_cli.representations.navigation import (
    error_neighborhood,
    error_neighborhood_data,
    target_timeline,
    target_timeline_data,
)
from self_improve_cli.representations.run_detail import run_detail, run_detail_data
from self_improve_cli.representations.skeleton import build_skeleton, skeleton_data
from self_improve_cli.representations.ter import write_ter
from self_improve_cli.representations.tools import (
    build_tools_detail,
    build_tools_overview,
)

__all__ = [
    "build_narrative",
    "build_skeleton",
    "build_tools_detail",
    "build_tools_overview",
    "context_at",
    "context_at_data",
    "error_neighborhood",
    "error_neighborhood_data",
    "narrative_data",
    "run_detail",
    "run_detail_data",
    "significant_runs",
    "skeleton_data",
    "target_timeline",
    "target_timeline_data",
    "write_ter",
]
