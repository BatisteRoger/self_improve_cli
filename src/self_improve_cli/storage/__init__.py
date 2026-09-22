"""Safe local storage for traces and derived artifacts.

By default, only sanitized traces are persisted. Raw retention requires an
explicit opt-in and is always local-only, never sent elsewhere.

Data layout (all under a configurable data root, gitignored):
- data/traces/<trace_id>/sanitized.json   — anonymized canonical trace
- data/traces/<trace_id>/raw.json         — raw trace (only if keep_raw=True)
- data/traces/<trace_id>/assessment.json  — manual task & outcome assessment
- data/traces/<trace_id>/findings.json    — persisted analyst findings
- data/ter/<trace_id>/                    — derived representations
- data/evaluations/<trace_id>/            — evaluations (versioned knowledge)

`TraceStore` is the single public entry point, composed from per-area
mixins: `traces` (sanitized/raw files), `records` (assessment + findings),
`ter` (derived representations), `prompts` (local prompt copies), `ati`
(target-agent architecture docs), over `base` (path safety + layout).
"""

from self_improve_cli.storage.ati import AtiMixin
from self_improve_cli.storage.base import _validate_id
from self_improve_cli.storage.prompts import PromptsMixin
from self_improve_cli.storage.records import RecordsMixin
from self_improve_cli.storage.ter import TerMixin
from self_improve_cli.storage.traces import TracesMixin

__all__ = ["TraceStore", "_validate_id"]


class TraceStore(TracesMixin, RecordsMixin, TerMixin, PromptsMixin, AtiMixin):
    """Safe local storage for traces and analyst artifacts.

    Composed from per-area mixins; see `storage/base.py` for the data-root
    layout and path-safety rules shared by all of them.
    """
