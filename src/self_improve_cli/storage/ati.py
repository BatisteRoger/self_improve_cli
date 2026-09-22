"""ATI documents — target agent architecture docs under data/ati/."""

from __future__ import annotations

from self_improve_cli.storage.base import StoreBase, _validate_id


class AtiMixin(StoreBase):
    """ATI (Agent-To-Improve) architecture document persistence."""

    def list_atis(self) -> list[str]:
        """List registered ATI (Agent-To-Improve) names."""
        ati_dir = self.data_root / "ati"
        if not ati_dir.exists():
            return []
        return sorted(
            d.name for d in ati_dir.iterdir() if d.is_dir() and (d / "architecture.md").exists()
        )

    def load_ati(self, name: str) -> str | None:
        """Load an ATI architecture document, or None if it doesn't exist."""
        _validate_id(name, "ati name")
        path = self.data_root / "ati" / name / "architecture.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
