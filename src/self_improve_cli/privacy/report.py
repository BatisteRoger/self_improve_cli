"""Sanitization report: non-sensitive summary of what was anonymized."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
