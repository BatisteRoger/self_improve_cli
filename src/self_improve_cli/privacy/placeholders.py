"""Stable, trace-local placeholder mapping.

The same original value always maps to the same placeholder within one
anonymization pass. The mapping is kept in memory only and never
persisted by default.
"""

from __future__ import annotations


class PlaceholderMap:
    """Stable, in-memory mapping from original values to typed placeholders."""

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
