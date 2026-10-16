"""Experiment receipts — `data/experiments/<session_id>/experiment.json`.

A receipt groups stored trace IDs by dataset example and records the
outcome policy and run conditions. Traces themselves stay in
`data/traces/`, so every trace view works on experiment traces unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from self_improve_cli.storage.base import StoreBase, _validate_id, write_json


class ExperimentsMixin(StoreBase):
    """Experiment receipt persistence and lookup."""

    @property
    def experiments_dir(self) -> Path:
        return self.data_root / "experiments"

    def save_experiment(self, receipt: dict[str, Any]) -> Path:
        session_id = _validate_id(receipt["session_id"], "session_id")
        path = self.experiments_dir / session_id / "experiment.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, receipt)
        return path

    def load_experiment(self, session_id: str) -> dict[str, Any] | None:
        path = self.experiments_dir / _validate_id(session_id, "session_id") / "experiment.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def find_experiment(self, name_or_id: str) -> dict[str, Any] | None:
        """Find a receipt by session ID first, then by exact experiment name."""
        try:
            found = self.load_experiment(name_or_id)
        except ValueError:
            found = None
        if found is not None:
            return found
        if not self.experiments_dir.exists():
            return None
        for path in sorted(self.experiments_dir.glob("*/experiment.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("experiment_name") == name_or_id:
                return data
        return None
