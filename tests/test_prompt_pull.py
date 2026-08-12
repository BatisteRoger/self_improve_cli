"""Tests for prompt pulling and manifest template extraction (offline, no network)."""

from __future__ import annotations

from self_improve_cli.sources.langsmith import _extract_template_from_manifest

# ---------------------------------------------------------------------------
# Synthetic manifest fixtures (mirrors LangSmith's serialized format)
# ---------------------------------------------------------------------------

_CHAT_PROMPT_MANIFEST = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "prompts", "chat", "ChatPromptTemplate"],
    "kwargs": {
        "messages": [
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "prompts", "chat", "SystemMessagePromptTemplate"],
                "kwargs": {
                    "prompt": {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
                        "kwargs": {
                            "input_variables": ["topic"],
                            "template_format": "mustache",
                            "template": "You are a helpful agent about {topic}.",
                        },
                    }
                },
            },
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "prompts", "chat", "HumanMessagePromptTemplate"],
                "kwargs": {
                    "prompt": {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
                        "kwargs": {
                            "input_variables": ["question"],
                            "template": "{question}",
                        },
                    }
                },
            },
        ]
    },
}

_PLAIN_PROMPT_MANIFEST = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
    "kwargs": {
        "input_variables": ["name"],
        "template": "Hello {name}, how are you?",
    },
}

_UNKNOWN_MANIFEST = {
    "lc": 1,
    "type": "constructor",
    "id": ["some", "unknown", "type"],
    "kwargs": {"custom": "data"},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_chat_prompt_template():
    """ChatPromptTemplate manifest: all message templates joined with newlines."""
    result = _extract_template_from_manifest(_CHAT_PROMPT_MANIFEST)
    assert "You are a helpful agent about {topic}." in result
    assert "{question}" in result
    # Both templates should be present, separated by a newline
    assert result.count("\n") >= 1


def test_extract_plain_prompt_template():
    """Plain PromptTemplate manifest: returns the single template string."""
    result = _extract_template_from_manifest(_PLAIN_PROMPT_MANIFEST)
    assert result == "Hello {name}, how are you?"


def test_extract_unknown_manifest_falls_back_to_json():
    """Unrecognized manifest structure: falls back to JSON serialization."""
    result = _extract_template_from_manifest(_UNKNOWN_MANIFEST)
    assert "unknown" in result
    assert "custom" in result
    # Should be valid JSON
    import json

    parsed = json.loads(result)
    assert parsed["kwargs"]["custom"] == "data"


def test_extract_non_dict_manifest():
    """Non-dict manifest: falls back to JSON serialization."""
    result = _extract_template_from_manifest("not a dict")
    assert "not a dict" in result


def test_extract_empty_messages():
    """ChatPromptTemplate with empty messages list: falls back to JSON."""
    manifest = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "prompts", "chat", "ChatPromptTemplate"],
        "kwargs": {"messages": []},
    }
    result = _extract_template_from_manifest(manifest)
    # No templates found, should fall back to JSON
    assert "ChatPromptTemplate" in result


def test_extract_message_without_prompt():
    """Message without a 'prompt' key: skipped gracefully."""
    manifest = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "prompts", "chat", "ChatPromptTemplate"],
        "kwargs": {
            "messages": [
                {"lc": 1, "type": "constructor", "kwargs": {}},
                {
                    "lc": 1,
                    "type": "constructor",
                    "kwargs": {
                        "prompt": {
                            "kwargs": {"template": "Only valid message."}
                        }
                    },
                },
            ]
        },
    }
    result = _extract_template_from_manifest(manifest)
    assert result == "Only valid message."
