"""Privacy layer: anonymize traces before persistence.

This package provides recursive anonymization of canonical Trace objects,
replacing PII and secrets with typed, trace-local stable placeholders.

Design:
- PII detection uses Presidio when available, with regex fallback for common
  patterns (email, phone, IBAN, credit card).
- Secret detection uses project-independent regex patterns for API keys,
  tokens, bearer credentials, private keys, and connection strings.
- Placeholders are stable within one anonymization pass: the same original
  value always maps to the same placeholder, so correlations are preserved
  without retaining the original.
- The original-to-placeholder mapping is kept in memory only and never
  persisted by default.
- Fail closed: if anonymization cannot run or is incomplete, the caller is
  responsible for refusing to persist unprotected data.

Backend selection (``anonymize_trace(trace, backend=...)``):
- "auto": use Presidio if installed, fall back to regex. Default.
- "presidio": force Presidio; if unavailable, fall back to regex and
  record an error in the report.
- "regex": skip Presidio entirely (faster, less comprehensive for PII).

Anonymization is defense in depth, not a guarantee that all sensitive values
are removed. Always review outputs before sharing or publishing.

Internal modules:
- ``patterns``: regex patterns and secret type classification.
- ``presidio_adapter``: Presidio NLP analyzer (cached singleton).
- ``placeholders``: stable, trace-local placeholder mapping.
- ``report``: non-sensitive sanitization summary.
- ``redact``: recursive redaction logic and backend merge.
"""

from __future__ import annotations

from typing import Literal

from self_improve_cli.domain import Trace
from self_improve_cli.privacy.placeholders import PlaceholderMap
from self_improve_cli.privacy.presidio_adapter import is_presidio_available
from self_improve_cli.privacy.redact import _redact_run
from self_improve_cli.privacy.report import SanitizationReport

AnonymizerBackend = Literal["auto", "presidio", "regex"]

__all__ = [
    "AnonymizerBackend",
    "SanitizationReport",
    "anonymize_trace",
    "is_presidio_available",
]


def anonymize_trace(
    trace: Trace, backend: AnonymizerBackend = "auto"
) -> tuple[Trace, SanitizationReport]:
    """Anonymize a Trace in place and return (trace, report).

    Replaces PII and secrets with stable, trace-local placeholders. The
    original-to-placeholder mapping is kept in memory only and never
    persisted by default.

    Returns a SanitizationReport with counts by entity type (no matched
    values). The ``recognizer_version`` field indicates which detector was
    used: "presidio+regex", "regex-fallback", or "regex-only".

    This is defense in depth, not a guarantee. Always review outputs before
    sharing or publishing.
    """
    pmap = PlaceholderMap()
    report = SanitizationReport()

    for run in trace.runs:
        _redact_run(run, pmap, report, backend)

    trace.sanitized = True
    trace.sanitization_report = report.to_dict()
    return trace, report
