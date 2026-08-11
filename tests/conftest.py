"""Synthetic test fixtures — fully fictional, no real data."""

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def load_mini_trace() -> list[dict]:
    """Load the synthetic mini trace fixture (raw dict format, for source adapter tests)."""
    return json.loads((FIXTURES_DIR / "mini_trace.json").read_text(encoding="utf-8"))
