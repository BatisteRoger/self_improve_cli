"""Sanitization report: non-sensitive summary of what was anonymized."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Version of the sanitization policy applied at fetch time, stored in each
# trace's sanitization report. Bump it whenever sanitization gets stricter:
# `self-improve doctor` then flags traces stored under an older version.
#   v2: structural redaction (value map, drop keys) and French PII patterns.
POLICY_VERSION = 2


@dataclass
class SanitizationReport:
    """Non-sensitive summary of what was anonymized.

    Never contains matched values — only counts by entity type.
    """

    entity_counts: dict[str, int] = field(default_factory=dict)
    recognizer_version: str = "regex-fallback"
    policy_version: str = str(POLICY_VERSION)
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
