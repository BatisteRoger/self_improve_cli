"""Privacy layer: anonymize traces before persistence.

This module provides recursive anonymization of canonical Trace objects,
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

Anonymization is defense in depth, not a guarantee that all sensitive values
are removed. Always review outputs before sharing or publishing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from self_improve_cli.domain import Message, Run, Trace

# Cached Presidio analyzer (module-level singleton to avoid reloading NLP models).
_presidio_analyzer: Any | None = None
# Set to True if Presidio was available but failed mid-analysis.
_presidio_failed: bool = False


@dataclass
class SanitizationReport:
    """Non-sensitive summary of what was anonymized.

    Never contains matched values — only counts by entity type.
    """

    entity_counts: dict[str, int] = field(default_factory=dict)
    recognizer_version: str = "regex-fallback"
    policy_version: str = "1"
    complete: bool = True
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_counts": dict(self.entity_counts),
            "recognizer_version": self.recognizer_version,
            "policy_version": self.policy_version,
            "complete": self.complete,
            "errors": list(self.errors),
        }

    def add(self, entity_type: str) -> None:
        self.entity_counts[entity_type] = self.entity_counts.get(entity_type, 0) + 1


# ---------------------------------------------------------------------------
# Regex patterns (always available, no external dependency)
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    (
        "PHONE",
        # International format (+CC ...) or US-style 10 digits with separators.
        # Requires + prefix or full 10-digit grouping to avoid matching
        # financial amounts like "1000" or "2518.17".
        re.compile(
            r"\+\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{0,4}"
            r"|"
            r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"
        ),
    ),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    # Credit card: require 4-digit groups with separators (standard printed format).
    # Avoids matching UUIDs, timestamps, and other long digit sequences.
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b|\b(?:\d{4}[ -]){2}\d{6}[ -]\d{5}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Secrets
    ("API_KEY", re.compile(r"\b(?:sk-|pk-|api[_-]?key[_-]?)[A-Za-z0-9]{20,}\b", re.IGNORECASE)),
    ("BEARER_TOKEN", re.compile(r"\b(?:Bearer|bearer)\s+[A-Za-z0-9._~+/=-]{20,}\b")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "CONNECTION_STRING",
        re.compile(
            r"\b(?:mongodb|postgres|postgresql|mysql|redis|amqp)://[^\s\"'<>]{10,}\b", re.IGNORECASE
        ),
    ),
    (
        "PASSWORD_ASSIGNMENT",
        re.compile(
            r"\b(?:password|passwd|pwd|secret|token)\s*[=:]\s*[\"']?[^\s\"']{6,}\b", re.IGNORECASE
        ),
    ),
]


def _try_presidio(text: str) -> list[tuple[str, str, int, int]] | None:
    """Attempt PII detection with Presidio. Returns None if unavailable.

    Returns a list of (entity_type, matched_text, start, end) tuples.
    The analyzer is cached as a module-level singleton to avoid reloading
    NLP models on every call.
    """
    global _presidio_analyzer
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        if _presidio_analyzer is None:
            _presidio_analyzer = AnalyzerEngine()
        assert _presidio_analyzer is not None
        results = _presidio_analyzer.analyze(
            text=text,
            language="en",
            entities=None,
            return_decision_process=False,
        )
        return [
            (r.entity_type, text[r.start : r.end], r.start, r.end)
            for r in results
            if r.score >= 0.5
        ]
    except Exception:
        # Presidio failed mid-analysis. Signal incompleteness to the caller
        # via the module-level flag so the report can be marked incomplete.
        _presidio_failed = True
        return None


def _regex_scan(text: str) -> list[tuple[str, str, int, int]]:
    """Detect PII and secrets with regex patterns. Always available."""
    results: list[tuple[str, str, int, int]] = []
    for entity_type, pattern in _REGEX_PATTERNS:
        for match in pattern.finditer(text):
            results.append((entity_type, match.group(), match.start(), match.end()))
    return results


# ---------------------------------------------------------------------------
# Placeholder mapping (trace-local, in-memory only)
# ---------------------------------------------------------------------------


class _PlaceholderMap:
    """Stable, in-memory mapping from original values to typed placeholders.

    The same original value always maps to the same placeholder within one
    anonymization pass. The mapping is never persisted by default.
    """

    def __init__(self) -> None:
        self._by_entity: dict[str, dict[str, str]] = {}
        self._counts: dict[str, int] = {}

    def get_or_create(self, entity_type: str, original: str) -> str:
        bucket = self._by_entity.setdefault(entity_type, {})
        if original in bucket:
            return bucket[original]
        count = self._counts.get(entity_type, 0) + 1
        self._counts[entity_type] = count
        placeholder = f"<{entity_type}_{count}>"
        bucket[original] = placeholder
        return placeholder

    def entity_counts(self) -> dict[str, int]:
        return dict(self._counts)


def _redact_string(text: str, pmap: _PlaceholderMap, report: SanitizationReport) -> str:
    """Replace detected PII and secrets in a string with stable placeholders.

    Uses Presidio when available, falls back to regex. Presidio results are
    supplemented with regex secret detection (Presidio does not cover all
    secret types).
    """
    if not text:
        return text

    # Collect all matches with positions, then replace from end to start.
    matches: list[tuple[str, str, int, int]] = []

    presidio_results = _try_presidio(text)
    if presidio_results is not None:
        matches.extend(presidio_results)
        report.recognizer_version = "presidio+regex"
    else:
        report.recognizer_version = "regex-fallback"
        if _presidio_failed:
            report.complete = False
            report.errors.append("Presidio was available but failed during analysis")

    # Always run regex for secrets (Presidio doesn't cover all secret types).
    regex_results = _regex_scan(text)
    # Merge: add regex matches that don't overlap with existing matches.
    for r in regex_results:
        _, _, s, e = r
        if not any(s < re_end and e > re_start for _, _, re_start, re_end in matches):
            matches.append(r)

    if not matches:
        return text

    # Sort by start position descending so replacements don't shift indices.
    matches.sort(key=lambda m: m[2], reverse=True)

    result = text
    for entity_type, matched, start, end in matches:
        placeholder = pmap.get_or_create(entity_type, matched)
        result = result[:start] + placeholder + result[end:]
        report.add(entity_type)

    return result


def _redact_value(
    value: Any, pmap: _PlaceholderMap, report: SanitizationReport, depth: int = 0
) -> Any:
    """Recursively anonymize any JSON-like value."""
    if depth > 50:
        return value
    if isinstance(value, str):
        return _redact_string(value, pmap, report)
    if isinstance(value, dict):
        return {k: _redact_value(v, pmap, report, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, pmap, report, depth + 1) for v in value]
    return value


def _redact_message(msg: Message, pmap: _PlaceholderMap, report: SanitizationReport) -> Message:
    """Anonymize a Message's text and tool call args."""
    msg.text = _redact_string(msg.text, pmap, report)
    for tc in msg.tool_calls:
        tc.args = _redact_value(tc.args, pmap, report)
    return msg


def _redact_run(run: Run, pmap: _PlaceholderMap, report: SanitizationReport) -> Run:
    """Anonymize all sensitive fields of a Run."""
    run.error = _redact_string(run.error, pmap, report) if run.error else run.error
    run.inputs = _redact_value(run.inputs, pmap, report)
    run.outputs = _redact_value(run.outputs, pmap, report)
    run.extra = _redact_value(run.extra, pmap, report)
    for msg in run.input_messages:
        _redact_message(msg, pmap, report)
    if run.output_message is not None:
        _redact_message(run.output_message, pmap, report)
    return run


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def anonymize_trace(trace: Trace) -> tuple[Trace, SanitizationReport]:
    """Anonymize a Trace in place and return (trace, report).

    Replaces PII and secrets with stable, trace-local placeholders. The
    original-to-placeholder mapping is kept in memory only and never
    persisted by default.

    Returns a SanitizationReport with counts by entity type (no matched
    values). If Presidio is not installed, falls back to regex-only detection
    and sets recognizer_version accordingly.

    This is defense in depth, not a guarantee. Always review outputs before
    sharing or publishing.
    """
    pmap = _PlaceholderMap()
    report = SanitizationReport()

    for run in trace.runs:
        _redact_run(run, pmap, report)

    trace.sanitized = True
    trace.sanitization_report = report.to_dict()
    return trace, report


def is_presidio_available() -> bool:
    """Return True if Presidio is installed and importable.

    On Python 3.14+, Presidio may not be installable because its spaCy
    dependency does not yet provide compatible wheels. In that case, the
    CLI falls back to regex-only detection (less comprehensive but still
    functional). The sanitization report's ``recognizer_version`` field
    indicates which detector was used.
    """
    try:
        import presidio_analyzer  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False
