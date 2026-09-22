"""Shared storage base — safe path handling and serialization helpers.

Trace IDs and filenames are used in path construction. Reject anything
that isn't a safe identifier to prevent path traversal (e.g. "../etc").
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from self_improve_cli.domain import RunType

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_id(value: str, label: str = "id") -> str:
    """Ensure a trace_id or filename is safe for path construction."""
    if not value or not _SAFE_ID.match(value):
        raise ValueError(
            f"Invalid {label}: {value!r}. Only letters, digits, dots, "
            "hyphens, and underscores are allowed."
        )
    return value


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to plain dicts for JSON."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, RunType):
        return obj.value
    return obj


def write_json(path: Path, payload: Any) -> Path:
    """Write a JSON payload to disk (UTF-8, indented)."""
    path.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


class StoreBase:
    """Data-root layout and path validation shared by all store mixins.

    Args:
        data_root: Root directory for local data. Defaults to ./data.
        keep_raw: If True, raw traces are also persisted (local-only, opt-in).
                  Never the default. Always accompanied by a warning.
    """

    def __init__(self, data_root: Path | None = None, keep_raw: bool = False) -> None:
        self.data_root = data_root or Path("data")
        self.traces_dir = self.data_root / "traces"
        self.ter_dir = self.data_root / "ter"
        self.evaluations_dir = self.data_root / "evaluations"
        self.keep_raw = keep_raw

    def _trace_dir(self, trace_id: str) -> Path:
        _validate_id(trace_id, "trace_id")
        return self.traces_dir / trace_id
