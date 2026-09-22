"""Derived TER artifacts — representation files written at fetch time."""

from __future__ import annotations

from pathlib import Path

from self_improve_cli.storage.base import StoreBase, _validate_id


class TerMixin(StoreBase):
    """Derived representation file persistence under data/ter/."""

    def save_ter_file(self, trace_id: str, filename: str, content: str) -> Path:
        """Write a derived representation file."""
        _validate_id(trace_id, "trace_id")
        _validate_id(filename, "filename")
        ter_dir = self.ter_dir / trace_id
        ter_dir.mkdir(parents=True, exist_ok=True)
        path = ter_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def load_ter_file(self, trace_id: str, filename: str) -> str | None:
        """Read a derived representation file, or None if it doesn't exist."""
        _validate_id(trace_id, "trace_id")
        _validate_id(filename, "filename")
        path = self.ter_dir / trace_id / filename
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
