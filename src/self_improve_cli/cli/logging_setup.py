"""Logging setup: INFO to console, DEBUG to data/self_improve.log."""

from __future__ import annotations

import logging
from pathlib import Path

_configured = False


def setup_logging(data_root: Path | None = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _configured
    if _configured:
        return

    data_root = data_root or Path("data")
    data_root.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    file_handler = logging.FileHandler(data_root / "self_improve.log", encoding="utf-8")
    # INFO level for the file log — DEBUG would capture full trace content
    # and SDK internals that may include sensitive data.
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    # Keep third-party HTTP noise out of the console and log file.
    # SDKs (LangSmith, urllib3, httpx) can log sensitive data at DEBUG level.
    for name in ("urllib3", "httpx", "httpcore", "langsmith", "openai", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
