"""Regex patterns for PII and secret detection.

Always available — no external dependencies. These patterns are run
regardless of the active backend, because they cover secret types that
Presidio does not, and because they are more specific for certain
overlapping cases (e.g. JWT vs Presidio's URL detector).
"""

from __future__ import annotations

import re

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
            r"\b(?:mongodb|postgres|postgresql|mysql|redis|amqp)://[^\s\"'<>]{10,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PASSWORD_ASSIGNMENT",
        re.compile(
            r"\b(?:password|passwd|pwd|secret|token)\s*[=:]\s*[\"']?[^\s\"']{6,}\b",
            re.IGNORECASE,
        ),
    ),
]

# Entity types detected by regex that are security-sensitive secrets.
# These take priority over Presidio matches when they overlap, because the
# regex patterns are more specific (e.g. Presidio may detect a JWT as a URL).
SECRET_TYPES = frozenset(
    {"API_KEY", "BEARER_TOKEN", "PRIVATE_KEY", "JWT", "CONNECTION_STRING", "PASSWORD_ASSIGNMENT"}
)

# Map Presidio entity names to the project's canonical names so that output
# is deterministic regardless of which backend (presidio vs regex-fallback) is
# active. Presidio uses different names for some entities (e.g. EMAIL_ADDRESS
# instead of EMAIL).
PRESIDIO_ENTITY_MAP: dict[str, str] = {
    "EMAIL_ADDRESS": "EMAIL",
    "IBAN_CODE": "IBAN",
}


def regex_scan(text: str) -> list[tuple[str, str, int, int]]:
    """Detect PII and secrets with regex patterns. Always available."""
    results: list[tuple[str, str, int, int]] = []
    for entity_type, pattern in _REGEX_PATTERNS:
        for match in pattern.finditer(text):
            results.append((entity_type, match.group(), match.start(), match.end()))
    return results
