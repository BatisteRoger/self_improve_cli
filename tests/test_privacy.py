"""Tests for the privacy/anonymization layer.

Detection tests are parametrized over both backends ("regex" and "presidio")
to ensure the package works with and without Presidio installed. Tests that
can't run in a given mode (e.g. presidio mode when presidio isn't installed)
are skipped automatically.
"""

import re

import pytest

from self_improve_cli.domain import Message, RunType, ToolCall, Trace
from self_improve_cli.privacy import anonymize_trace, is_presidio_available
from tests.helpers import make_root, make_run

# Parametrize detection tests over both backends.
# In "presidio" mode, skip if Presidio isn't installed.
_BACKENDS = ["regex", "presidio"]


def _skip_if_presidio_unavailable(backend: str) -> None:
    if backend == "presidio" and not is_presidio_available():
        pytest.skip("Presidio not installed — skipping presidio-backend test")


def _trace_with_text(*texts: str) -> Trace:
    """Build a minimal trace with the given strings in run inputs."""
    runs = [make_root()]
    for i, text in enumerate(texts):
        runs.append(
            make_run(
                f"r{i + 1}",
                run_type=RunType.LLM,
                inputs={"content": text},
                input_messages=[Message(role="human", text=text)],
            )
        )
    return Trace(trace_id="t1", runs=runs)


# ---------------------------------------------------------------------------
# Detection tests (parametrized over both backends)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_replaces_email(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("Contact me at alice@example.com please.")
    anonymized, report = anonymize_trace(trace, backend=backend)
    assert trace.sanitized
    assert "alice@example.com" not in str(anonymized.runs[1].inputs)
    assert "<EMAIL_" in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("EMAIL", 0) >= 1


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_replaces_phone(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("Call +33 6 12 34 56 78 tomorrow.")
    anonymized, report = anonymize_trace(trace, backend=backend)
    assert "+33 6 12 34 56 78" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("PHONE", 0) >= 1


def test_phone_regex_does_not_match_financial_amounts():
    """Financial amounts like 1000, 2518.17 should NOT be detected as phone numbers."""
    trace = _trace_with_text("Initial capital: 1000 euros. Final value: 2518.17 euros.")
    _, report = anonymize_trace(trace, backend="regex")
    assert report.entity_counts.get("PHONE", 0) == 0


def test_phone_regex_matches_us_format():
    """US-style 10-digit numbers with separators should still be detected."""
    trace = _trace_with_text("Call me at 555-123-4567.")
    anonymized, report = anonymize_trace(trace, backend="regex")
    assert "555-123-4567" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("PHONE", 0) >= 1


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_replaces_iban(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("My IBAN is FR1420041010050500013M02606.")
    anonymized, report = anonymize_trace(trace, backend=backend)
    assert "FR1420041010050500013M02606" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("IBAN", 0) >= 1


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_replaces_credit_card(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("Card: 4111 1111 1111 1111")
    anonymized, report = anonymize_trace(trace, backend=backend)
    assert "4111 1111 1111 1111" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("CREDIT_CARD", 0) >= 1


def test_credit_card_regex_does_not_match_uuids():
    """UUIDs and other long digit sequences should NOT be detected as credit cards."""
    trace = _trace_with_text("trace id: 019fd2411b2977e8b8c16a7935085aae")
    _, report = anonymize_trace(trace, backend="regex")
    assert report.entity_counts.get("CREDIT_CARD", 0) == 0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_replaces_api_key(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("Use key sk-abcdefghijklmnopqrstuvwxyz123456 for auth.")
    anonymized, report = anonymize_trace(trace, backend=backend)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("API_KEY", 0) >= 1


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_replaces_bearer_token(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text(
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
    )
    _, report = anonymize_trace(trace, backend=backend)
    # JWT should be caught (regex takes priority over Presidio's URL detector)
    assert (
        report.entity_counts.get("JWT", 0) >= 1
        or report.entity_counts.get("BEARER_TOKEN", 0) >= 1
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_replaces_private_key(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...")
    anonymized, report = anonymize_trace(trace, backend=backend)
    assert "BEGIN RSA PRIVATE KEY" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("PRIVATE_KEY", 0) >= 1


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_stable_placeholders_within_trace(backend):
    """Same email in two places should get the same placeholder."""
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text(
        "Email alice@example.com here.",
        "Also alice@example.com there.",
    )
    anonymized, _ = anonymize_trace(trace, backend=backend)
    text1 = str(anonymized.runs[1].inputs)
    text2 = str(anonymized.runs[2].inputs)
    placeholders1 = re.findall(r"<EMAIL_\d+>", text1)
    placeholders2 = re.findall(r"<EMAIL_\d+>", text2)
    assert len(placeholders1) == 1
    assert len(placeholders2) == 1
    assert placeholders1[0] == placeholders2[0]


# ---------------------------------------------------------------------------
# Structure tests (backend-agnostic — test recursion/dispatch, not detection)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_preserves_non_sensitive_text(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("What is 2+2? Use the calculator tool.")
    anonymized, report = anonymize_trace(trace, backend=backend)
    assert "What is 2+2?" in anonymized.runs[1].inputs["content"]
    assert report.entity_counts == {} or all(v == 0 for v in report.entity_counts.values())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_recursive_dict(backend):
    _skip_if_presidio_unavailable(backend)
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(),
            make_run("r1", inputs={"nested": {"deep": "alice@example.com"}}),
        ],
    )
    anonymized, _ = anonymize_trace(trace, backend=backend)
    assert "alice@example.com" not in str(anonymized.runs[1].inputs)
    assert "<EMAIL_" in str(anonymized.runs[1].inputs)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_recursive_list(backend):
    _skip_if_presidio_unavailable(backend)
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(),
            make_run("r1", inputs={"items": ["alice@example.com", "safe text"]}),
        ],
    )
    anonymized, _ = anonymize_trace(trace, backend=backend)
    assert "alice@example.com" not in str(anonymized.runs[1].inputs)
    assert "safe text" in anonymized.runs[1].inputs["items"][1]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_message_text(backend):
    _skip_if_presidio_unavailable(backend)
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(),
            make_run(
                "r1",
                input_messages=[Message(role="human", text="My email is bob@test.org")],
            ),
        ],
    )
    anonymized, _ = anonymize_trace(trace, backend=backend)
    assert "bob@test.org" not in anonymized.runs[1].input_messages[0].text
    assert "<EMAIL_" in anonymized.runs[1].input_messages[0].text


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_tool_call_args(backend):
    _skip_if_presidio_unavailable(backend)
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(),
            make_run(
                "r1",
                run_type=RunType.LLM,
                output_message=Message(
                    role="ai",
                    tool_calls=[ToolCall(name="send_email", args={"to": "alice@example.com"})],
                ),
            ),
        ],
    )
    anonymized, _ = anonymize_trace(trace, backend=backend)
    assert "alice@example.com" not in str(anonymized.runs[1].output_message.tool_calls[0].args)
    assert "<EMAIL_" in str(anonymized.runs[1].output_message.tool_calls[0].args)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_error_field(backend):
    _skip_if_presidio_unavailable(backend)
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(),
            make_run("r1", error="Failed to reach alice@example.com"),
        ],
    )
    anonymized, _ = anonymize_trace(trace, backend=backend)
    assert "alice@example.com" not in anonymized.runs[1].error
    assert "<EMAIL_" in anonymized.runs[1].error


# ---------------------------------------------------------------------------
# Report safety tests (backend-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_sanitization_report_no_matched_values(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("Email alice@example.com here.")
    _, report = anonymize_trace(trace, backend=backend)
    report_dict = report.to_dict()
    assert "alice@example.com" not in str(report_dict)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_empty_trace(backend):
    _skip_if_presidio_unavailable(backend)
    trace = Trace(trace_id="t1", runs=[])
    anonymized, report = anonymize_trace(trace, backend=backend)
    assert anonymized.sanitized
    assert report.entity_counts == {}


@pytest.mark.parametrize("backend", _BACKENDS)
def test_anonymize_no_secrets_in_report(backend):
    _skip_if_presidio_unavailable(backend)
    trace = _trace_with_text("key sk-abcdefghijklmnopqrstuvwxyz123456")
    _, report = anonymize_trace(trace, backend=backend)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(report.to_dict())


# ---------------------------------------------------------------------------
# Backend selection tests
# ---------------------------------------------------------------------------


def test_regex_backend_sets_recognizer_version():
    """backend='regex' produces recognizer_version='regex-only'."""
    trace = _trace_with_text("Email alice@example.com here.")
    _, report = anonymize_trace(trace, backend="regex")
    assert report.recognizer_version == "regex-only"


def test_auto_backend_sets_recognizer_version():
    """backend='auto' uses presidio+regex or regex-fallback depending on availability."""
    trace = _trace_with_text("Email alice@example.com here.")
    _, report = anonymize_trace(trace, backend="auto")
    if is_presidio_available():
        assert report.recognizer_version == "presidio+regex"
    else:
        assert report.recognizer_version == "regex-fallback"


def test_presidio_backend_falls_back_with_error_when_unavailable(monkeypatch):
    """backend='presidio' records an error when Presidio is not installed."""
    # Force try_presidio to return None as if not installed
    monkeypatch.setattr("self_improve_cli.privacy.redact.try_presidio", lambda text: None)
    trace = _trace_with_text("Email alice@example.com here.")
    _, report = anonymize_trace(trace, backend="presidio")
    assert report.recognizer_version == "regex-fallback"
    assert not report.complete
    assert any("not installed" in e for e in report.errors)


def test_regex_backend_skips_presidio_call(monkeypatch):
    """backend='regex' never calls try_presidio."""
    call_count = 0

    def _fail_if_called(text):
        nonlocal call_count
        call_count += 1
        return []

    monkeypatch.setattr("self_improve_cli.privacy.redact.try_presidio", _fail_if_called)
    trace = _trace_with_text("Email alice@example.com here.")
    anonymize_trace(trace, backend="regex")
    assert call_count == 0


def test_presidio_available_returns_bool():
    assert isinstance(is_presidio_available(), bool)
