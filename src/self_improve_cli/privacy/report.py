"""Sanitization report: non-sensitive summary of what was anonymized."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Version of the sanitization policy applied at fetch time, stored in each
# trace's sanitization report. Bump it whenever sanitization gets stricter:
# `self-improve doctor` then flags traces stored under an older version.
#   v2: structural redaction (value map, drop keys) and French PII patterns.
POLICY_VERSION = 2


def hardening_fingerprint(
    engine: str, drop_keys: Iterable[str], value_maps: Iterable[str]
) -> dict[str, Any]:
    """Summarize the hardening configuration applied to a trace.

    ``value_maps`` are the ``<table_key>:<value_field>:<token_field>`` specs.
    Only the engine and the counts are stored in clear. The configured key
    names enter a short hash, so a renamed key changes the fingerprint
    without the names being written to disk. No trace value is involved.
    """
    keys = sorted(set(drop_keys))
    maps = sorted(set(value_maps))
    material = "\n".join([engine, *keys, "--", *maps])
    return {
        "engine": engine,
        "drop_keys": len(keys),
        "value_maps": len(maps),
        "hash": hashlib.sha256(material.encode("utf-8")).hexdigest()[:12],
    }


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
    hardening: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_counts": dict(self.entity_counts),
            "recognizer_version": self.recognizer_version,
            "policy_version": self.policy_version,
            "complete": self.complete,
            "errors": list(self.errors),
            "hardening": dict(self.hardening) if self.hardening is not None else None,
        }

    def add(self, entity_type: str) -> None:
        self.entity_counts[entity_type] = self.entity_counts.get(entity_type, 0) + 1
