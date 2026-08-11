"""Tests for the privacy/anonymization layer."""

from self_improve_cli.domain import Message, RunType, ToolCall, Trace
from self_improve_cli.privacy import anonymize_trace, is_presidio_available
from tests.helpers import make_root, make_run


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


def test_anonymize_replaces_email():
    trace = _trace_with_text("Contact me at alice@example.com please.")
    anonymized, report = anonymize_trace(trace)
    assert trace.sanitized
    assert "alice@example.com" not in str(anonymized.runs[1].inputs)
    assert "<EMAIL_" in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("EMAIL", 0) >= 1


def test_anonymize_replaces_phone():
    trace = _trace_with_text("Call +33 6 12 34 56 78 tomorrow.")
    anonymized, report = anonymize_trace(trace)
    assert "+33 6 12 34 56 78" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("PHONE", 0) >= 1


def test_phone_regex_does_not_match_financial_amounts():
    """Financial amounts like 1000, 2518.17 should NOT be detected as phone numbers."""
    trace = _trace_with_text("Initial capital: 1000 euros. Final value: 2518.17 euros.")
    anonymized, report = anonymize_trace(trace)
    assert report.entity_counts.get("PHONE", 0) == 0


def test_phone_regex_matches_us_format():
    """US-style 10-digit numbers with separators should still be detected."""
    trace = _trace_with_text("Call me at 555-123-4567.")
    anonymized, report = anonymize_trace(trace)
    assert "555-123-4567" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("PHONE", 0) >= 1


def test_anonymize_replaces_iban():
    trace = _trace_with_text("My IBAN is FR1420041010050500013M02606.")
    anonymized, report = anonymize_trace(trace)
    assert "FR1420041010050500013M02606" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("IBAN", 0) >= 1


def test_anonymize_replaces_credit_card():
    trace = _trace_with_text("Card: 4111 1111 1111 1111")
    anonymized, report = anonymize_trace(trace)
    assert "4111 1111 1111 1111" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("CREDIT_CARD", 0) >= 1


def test_credit_card_regex_does_not_match_uuids():
    """UUIDs and other long digit sequences should NOT be detected as credit cards."""
    trace = _trace_with_text("trace id: 019fd2411b2977e8b8c16a7935085aae")
    anonymized, report = anonymize_trace(trace)
    assert report.entity_counts.get("CREDIT_CARD", 0) == 0


def test_anonymize_replaces_api_key():
    trace = _trace_with_text("Use key sk-abcdefghijklmnopqrstuvwxyz123456 for auth.")
    anonymized, report = anonymize_trace(trace)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("API_KEY", 0) >= 1


def test_anonymize_replaces_bearer_token():
    trace = _trace_with_text(
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
    )
    anonymized, report = anonymize_trace(trace)
    # JWT should be caught
    assert (
        report.entity_counts.get("JWT", 0) >= 1 or report.entity_counts.get("BEARER_TOKEN", 0) >= 1
    )


def test_anonymize_replaces_private_key():
    trace = _trace_with_text("-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...")
    anonymized, report = anonymize_trace(trace)
    assert "BEGIN RSA PRIVATE KEY" not in str(anonymized.runs[1].inputs)
    assert report.entity_counts.get("PRIVATE_KEY", 0) >= 1


def test_anonymize_stable_placeholders_within_trace():
    """Same email in two places should get the same placeholder."""
    trace = _trace_with_text(
        "Email alice@example.com here.",
        "Also alice@example.com there.",
    )
    anonymized, report = anonymize_trace(trace)
    text1 = str(anonymized.runs[1].inputs)
    text2 = str(anonymized.runs[2].inputs)
    # Both should contain the same placeholder
    import re

    placeholders1 = re.findall(r"<EMAIL_\d+>", text1)
    placeholders2 = re.findall(r"<EMAIL_\d+>", text2)
    assert len(placeholders1) == 1
    assert len(placeholders2) == 1
    assert placeholders1[0] == placeholders2[0]


def test_anonymize_preserves_non_sensitive_text():
    trace = _trace_with_text("What is 2+2? Use the calculator tool.")
    anonymized, report = anonymize_trace(trace)
    assert "What is 2+2?" in anonymized.runs[1].inputs["content"]
    assert report.entity_counts == {} or all(v == 0 for v in report.entity_counts.values())


def test_anonymize_recursive_dict():
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(),
            make_run("r1", inputs={"nested": {"deep": "alice@example.com"}}),
        ],
    )
    anonymized, report = anonymize_trace(trace)
    assert "alice@example.com" not in str(anonymized.runs[1].inputs)
    assert "<EMAIL_" in str(anonymized.runs[1].inputs)


def test_anonymize_recursive_list():
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(),
            make_run("r1", inputs={"items": ["alice@example.com", "safe text"]}),
        ],
    )
    anonymized, report = anonymize_trace(trace)
    assert "alice@example.com" not in str(anonymized.runs[1].inputs)
    assert "safe text" in anonymized.runs[1].inputs["items"][1]


def test_anonymize_message_text():
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
    anonymized, report = anonymize_trace(trace)
    assert "bob@test.org" not in anonymized.runs[1].input_messages[0].text
    assert "<EMAIL_" in anonymized.runs[1].input_messages[0].text


def test_anonymize_tool_call_args():
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
    anonymized, report = anonymize_trace(trace)
    assert "alice@example.com" not in str(anonymized.runs[1].output_message.tool_calls[0].args)
    assert "<EMAIL_" in str(anonymized.runs[1].output_message.tool_calls[0].args)


def test_anonymize_error_field():
    trace = Trace(
        trace_id="t1",
        runs=[
            make_root(),
            make_run("r1", error="Failed to reach alice@example.com"),
        ],
    )
    anonymized, report = anonymize_trace(trace)
    assert "alice@example.com" not in anonymized.runs[1].error
    assert "<EMAIL_" in anonymized.runs[1].error


def test_sanitization_report_no_matched_values():
    trace = _trace_with_text("Email alice@example.com here.")
    _, report = anonymize_trace(trace)
    report_dict = report.to_dict()
    # The report should not contain the original email
    assert "alice@example.com" not in str(report_dict)


def test_anonymize_empty_trace():
    trace = Trace(trace_id="t1", runs=[])
    anonymized, report = anonymize_trace(trace)
    assert anonymized.sanitized
    assert report.entity_counts == {}


def test_anonymize_no_secrets_in_report():
    trace = _trace_with_text("key sk-abcdefghijklmnopqrstuvwxyz123456")
    _, report = anonymize_trace(trace)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(report.to_dict())


def test_presidio_available_returns_bool():
    assert isinstance(is_presidio_available(), bool)
