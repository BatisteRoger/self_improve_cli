"""Redaction logic: apply detected entities to strings, values, messages, runs.

This module handles the recursive traversal of Trace objects and the
merge logic between regex and Presidio detection results. The actual
detection is delegated to ``patterns`` (regex) and ``presidio_adapter``
(Presidio).
"""

from __future__ import annotations

from typing import Any, Literal

from self_improve_cli.domain import Message, Run
from self_improve_cli.privacy.patterns import SECRET_TYPES, regex_scan
from self_improve_cli.privacy.placeholders import PlaceholderMap
from self_improve_cli.privacy.presidio_adapter import _presidio_failed, try_presidio
from self_improve_cli.privacy.report import SanitizationReport

AnonymizerBackend = Literal["auto", "presidio", "regex"]


def _redact_string(
    text: str,
    pmap: PlaceholderMap,
    report: SanitizationReport,
    backend: AnonymizerBackend = "auto",
) -> str:
    """Replace detected PII and secrets in a string with stable placeholders.

    Backend selection:
    - "auto": use Presidio if installed, fall back to regex. Default.
    - "presidio": force Presidio; if unavailable, fall back to regex and
      record an error in the report.
    - "regex": skip Presidio entirely (faster, less comprehensive for PII).

    Regex secret matches always take priority over Presidio when they overlap,
    because the regex patterns are more specific (e.g. Presidio may detect a
    JWT as a URL).
    """
    if not text:
        return text

    matches: list[tuple[str, str, int, int]] = []

    # 1. Run regex secrets first — they take priority over Presidio.
    regex_results = regex_scan(text)
    regex_secrets = [r for r in regex_results if r[0] in SECRET_TYPES]
    regex_pii = [r for r in regex_results if r[0] not in SECRET_TYPES]
    matches.extend(regex_secrets)

    # 2. Run Presidio (unless regex-only), add non-overlapping matches.
    use_presidio = backend in ("auto", "presidio")
    if use_presidio:
        presidio_results = try_presidio(text)
        if presidio_results is not None:
            report.recognizer_version = "presidio+regex"
            for r in presidio_results:
                _, _, s, e = r
                if not any(s < re_end and e > re_start for _, _, re_start, re_end in matches):
                    matches.append(r)
        elif backend == "presidio":
            report.recognizer_version = "regex-fallback"
            report.complete = False
            report.errors.append("Presidio backend requested but not installed")
        else:
            report.recognizer_version = "regex-fallback"
            if _presidio_failed:
                report.complete = False
                report.errors.append("Presidio was available but failed during analysis")
    else:
        report.recognizer_version = "regex-only"

    # 3. Add regex PII that don't overlap with existing matches.
    for r in regex_pii:
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
    value: Any,
    pmap: PlaceholderMap,
    report: SanitizationReport,
    depth: int = 0,
    backend: AnonymizerBackend = "auto",
) -> Any:
    """Recursively anonymize any JSON-like value."""
    if depth > 50:
        return value
    if isinstance(value, str):
        return _redact_string(value, pmap, report, backend)
    if isinstance(value, dict):
        return {k: _redact_value(v, pmap, report, depth + 1, backend) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, pmap, report, depth + 1, backend) for v in value]
    return value


def _redact_message(
    msg: Message,
    pmap: PlaceholderMap,
    report: SanitizationReport,
    backend: AnonymizerBackend = "auto",
) -> Message:
    """Anonymize a Message's text and tool call args."""
    msg.text = _redact_string(msg.text, pmap, report, backend)
    for tc in msg.tool_calls:
        tc.args = _redact_value(tc.args, pmap, report, backend=backend)
    return msg


def _redact_run(
    run: Run,
    pmap: PlaceholderMap,
    report: SanitizationReport,
    backend: AnonymizerBackend = "auto",
) -> Run:
    """Anonymize all sensitive fields of a Run."""
    run.error = _redact_string(run.error, pmap, report, backend) if run.error else run.error
    run.inputs = _redact_value(run.inputs, pmap, report, backend=backend)
    run.outputs = _redact_value(run.outputs, pmap, report, backend=backend)
    run.extra = _redact_value(run.extra, pmap, report, backend=backend)
    for msg in run.input_messages:
        _redact_message(msg, pmap, report, backend)
    if run.output_message is not None:
        _redact_message(run.output_message, pmap, report, backend)
    return run
