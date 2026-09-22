"""Trace files — sanitized and raw trace persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from self_improve_cli.domain import (
    Message,
    Run,
    RunType,
    ToolCall,
    Trace,
)
from self_improve_cli.storage.base import (
    StoreBase,
    _dataclass_to_dict,
    write_json,
)

logger = logging.getLogger(__name__)


def _dict_to_trace(data: dict[str, Any]) -> Trace:
    """Reconstruct a Trace from a JSON-deserialized dict."""
    runs = []
    for run_data in data.get("runs", []):
        input_messages = [
            Message(
                role=m.get("role", "?"),
                text=m.get("text", ""),
                tool_calls=[
                    ToolCall(
                        name=tc.get("name", "?"),
                        args=tc.get("args", {}),
                        id=tc.get("id"),
                    )
                    for tc in m.get("tool_calls", [])
                ],
                tool_call_id=m.get("tool_call_id"),
            )
            for m in run_data.get("input_messages", [])
        ]
        output_msg_data = run_data.get("output_message")
        output_message = (
            Message(
                role=output_msg_data.get("role", "?"),
                text=output_msg_data.get("text", ""),
                tool_calls=[
                    ToolCall(
                        name=tc.get("name", "?"),
                        args=tc.get("args", {}),
                        id=tc.get("id"),
                    )
                    for tc in output_msg_data.get("tool_calls", [])
                ],
                tool_call_id=output_msg_data.get("tool_call_id"),
            )
            if output_msg_data
            else None
        )
        runs.append(
            Run(
                id=run_data.get("id", ""),
                trace_id=run_data.get("trace_id", ""),
                run_type=RunType(run_data.get("run_type", "other")),
                name=run_data.get("name", ""),
                parent_run_id=run_data.get("parent_run_id"),
                dotted_order=run_data.get("dotted_order"),
                status=run_data.get("status"),
                start_time=run_data.get("start_time"),
                end_time=run_data.get("end_time"),
                total_tokens=run_data.get("total_tokens"),
                prompt_tokens=run_data.get("prompt_tokens"),
                completion_tokens=run_data.get("completion_tokens"),
                error=run_data.get("error"),
                inputs=run_data.get("inputs", {}),
                outputs=run_data.get("outputs", {}),
                input_messages=input_messages,
                output_message=output_message,
                extra=run_data.get("extra", {}),
            )
        )
    return Trace(
        trace_id=data.get("trace_id", ""),
        runs=runs,
        schema_version=data.get("schema_version", 1),
        sanitized=data.get("sanitized", False),
        sanitization_report=data.get("sanitization_report"),
        source=data.get("source"),
        extra=data.get("extra", {}),
    )


class TracesMixin(StoreBase):
    """Sanitized/raw trace file persistence."""

    def save_sanitized(self, trace: Trace) -> Path:
        """Persist a sanitized trace as JSON. Returns the path."""
        if not trace.sanitized:
            raise ValueError(
                "Refusing to persist a non-sanitized trace. "
                "Call anonymize_trace() first, or use save_raw() with explicit opt-in."
            )
        trace_dir = self._trace_dir(trace.trace_id)
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / "sanitized.json"
        payload = _dataclass_to_dict(trace)
        write_json(path, payload)
        logger.info(
            "Saved sanitized trace %s to %s (%.1f KB)",
            trace.trace_id,
            path,
            path.stat().st_size / 1024,
        )
        return path

    def save_raw(self, trace: Trace) -> Path | None:
        """Persist a raw trace as JSON. Only if keep_raw is True.

        This is an explicit opt-in. Always logs a warning.
        """
        if not self.keep_raw:
            logger.warning("save_raw called but keep_raw=False. Skipping.")
            return None
        trace_dir = self._trace_dir(trace.trace_id)
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / "raw.json"
        payload = _dataclass_to_dict(trace)
        write_json(path, payload)
        logger.warning(
            "Saved RAW trace %s to %s — this file contains unanonymized data. "
            "Do NOT commit or share it.",
            trace.trace_id,
            path,
        )
        return path

    def load_trace(self, trace_id: str) -> Trace:
        """Load a previously saved trace from disk.

        Prefers sanitized.json. Falls back to raw.json only if it exists
        (and logs a warning). Raises ValueError if the loaded trace is not
        sanitized — callers must check before displaying content.
        """
        sanitized_path = self._trace_dir(trace_id) / "sanitized.json"
        if sanitized_path.exists():
            data = json.loads(sanitized_path.read_text(encoding="utf-8"))
            return _dict_to_trace(data)

        raw_path = self._trace_dir(trace_id) / "raw.json"
        if raw_path.exists():
            logger.warning("Loading RAW trace from %s — this data is not anonymized.", raw_path)
            data = json.loads(raw_path.read_text(encoding="utf-8"))
            trace = _dict_to_trace(data)
            if not trace.sanitized:
                raise ValueError(
                    f"Trace {trace_id} was loaded from raw.json and is NOT sanitized. "
                    "This data may contain PII and secrets. Re-fetch with `self-improve fetch` "
                    "to produce a sanitized copy, or handle with extreme care."
                )
            return trace

        raise FileNotFoundError(f"No saved trace at {self._trace_dir(trace_id)}")

    def trace_exists(self, trace_id: str) -> bool:
        """Return True if a saved trace (sanitized or raw) exists."""
        trace_dir = self._trace_dir(trace_id)
        return (trace_dir / "sanitized.json").exists() or (trace_dir / "raw.json").exists()
