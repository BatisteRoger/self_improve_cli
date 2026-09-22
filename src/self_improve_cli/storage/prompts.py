"""Local prompt copies — templates pulled from a prompt hub."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from self_improve_cli.storage.base import StoreBase, _validate_id

logger = logging.getLogger(__name__)


class PromptsMixin(StoreBase):
    """Local prompt file persistence under data/prompts/."""

    def save_prompt(self, name: str, tag: str, content: str) -> Path:
        """Save a prompt template as a local .md file.

        Args:
            name: Prompt name (e.g. "react_agent").
            tag: Tag or "latest".
            content: The prompt template text.
        """
        _validate_id(name, "prompt name")
        _validate_id(tag, "tag")
        prompt_dir = self.data_root / "prompts" / name
        prompt_dir.mkdir(parents=True, exist_ok=True)
        path = prompt_dir / f"{tag}.md"
        path.write_text(content, encoding="utf-8")
        logger.info("Saved prompt %s:%s to %s", name, tag, path)
        return path

    def load_prompt(self, name: str, tag: str = "latest") -> str | None:
        """Load a locally saved prompt, or None if it doesn't exist."""
        _validate_id(name, "prompt name")
        _validate_id(tag, "tag")
        path = self.data_root / "prompts" / name / f"{tag}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_prompts(self) -> list[dict[str, Any]]:
        """List all locally saved prompts.

        Returns a list of dicts with name, tag, size, and modified time.
        """
        prompts_dir = self.data_root / "prompts"
        if not prompts_dir.exists():
            return []
        result = []
        for name_dir in sorted(prompts_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            for md_file in sorted(name_dir.glob("*.md")):
                stat = md_file.stat()
                result.append(
                    {
                        "name": name_dir.name,
                        "tag": md_file.stem,
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )
        return result
