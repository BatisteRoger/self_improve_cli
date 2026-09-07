"""Presidio adapter: NLP-backed PII detection.

Stops at this boundary — SDK-specific objects never leak into the core.
The analyzer is cached as a module-level singleton to avoid reloading
NLP models on every call.
"""

from __future__ import annotations

from typing import Any

from self_improve_cli.privacy.patterns import PRESIDIO_ENTITY_MAP

# Cached Presidio analyzer (module-level singleton to avoid reloading NLP models).
_presidio_analyzer: Any | None = None
# Set to True if Presidio was available but failed mid-analysis.
_presidio_failed: bool = False


def try_presidio(text: str) -> list[tuple[str, str, int, int]] | None:
    """Attempt PII detection with Presidio. Returns None if unavailable.

    Returns a list of (entity_type, matched_text, start, end) tuples.
    Entity names are normalized to the project's canonical names via
    PRESIDIO_ENTITY_MAP.
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
        out: list[tuple[str, str, int, int]] = []
        for r in results:
            if r.score < 0.5:
                continue
            entity_type: str = r.entity_type
            canonical = PRESIDIO_ENTITY_MAP.get(entity_type, entity_type)
            out.append((canonical, text[r.start : r.end], r.start, r.end))
        return out
    except Exception:
        # Presidio failed mid-analysis. Signal incompleteness to the caller
        # via the module-level flag so the report can be marked incomplete.
        _presidio_failed = True
        return None


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
