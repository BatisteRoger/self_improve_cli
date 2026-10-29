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
- ``structural``: mapping-table value replacement and key dropping.
"""

from __future__ import annotations

from typing import Any, Literal

from self_improve_cli.domain import Trace
from self_improve_cli.privacy.placeholders import PlaceholderMap
from self_improve_cli.privacy.presidio_adapter import is_presidio_available
from self_improve_cli.privacy.redact import _redact_run
from self_improve_cli.privacy.report import SanitizationReport, hardening_fingerprint
from self_improve_cli.privacy.structural import (
    ValueMapSpec,
    collect_value_map,
    parse_drop_keys,
    parse_value_maps,
    replace_values,
)
from self_improve_cli.privacy.structural import drop_keys as _drop_keys

AnonymizerBackend = Literal["auto", "presidio", "regex"]

__all__ = [
    "AnonymizerBackend",
    "SanitizationReport",
    "ValueMapSpec",
    "anonymize_trace",
    "hardening_for",
    "is_presidio_available",
    "parse_drop_keys",
    "parse_value_maps",
]


def hardening_for(
    backend: str, drop_keys: frozenset[str], value_maps: tuple[ValueMapSpec, ...]
) -> dict[str, Any]:
    """Fingerprint of a hardening configuration, as stored in the report."""
    specs = (f"{s.table_key}:{s.value_field}:{s.token_field}" for s in value_maps)
    return hardening_fingerprint(backend, drop_keys, specs)


def anonymize_trace(
    trace: Trace,
    backend: AnonymizerBackend = "auto",
    drop_keys: frozenset[str] = frozenset(),
    value_maps: tuple[ValueMapSpec, ...] = (),
) -> tuple[Trace, SanitizationReport]:
    """Anonymize a Trace in place and return (trace, report).

    Replaces PII and secrets with stable, trace-local placeholders. The
    original-to-placeholder mapping is kept in memory only and never
    persisted by default.

    Before pattern detection, two structural passes run (see ``structural``):
    values listed in the ``value_maps`` tables are replaced by their tokens
    everywhere in the trace, then every ``drop_keys`` key and every table key
    is replaced with a ``<dropped:N chars>`` marker.

    Returns a SanitizationReport with counts by entity type (no matched
    values). The ``recognizer_version`` field indicates which detector was
    used: "presidio+regex", "regex-fallback", or "regex-only".

    This is defense in depth, not a guarantee. Always review outputs before
    sharing or publishing.
    """
    pmap = PlaceholderMap()
    report = SanitizationReport(hardening=hardening_for(backend, drop_keys, value_maps))

    mapping = collect_value_map(trace.runs, value_maps)
    if replaced := replace_values(trace.runs, mapping):
        report.entity_counts["MAPPED_VALUE"] = replaced
    keys = drop_keys | {spec.table_key for spec in value_maps}
    if dropped := _drop_keys(trace.runs, keys):
        report.entity_counts["DROPPED_KEY"] = dropped

    for run in trace.runs:
        _redact_run(run, pmap, report, backend)

    trace.sanitized = True
    trace.sanitization_report = report.to_dict()
    return trace, report
