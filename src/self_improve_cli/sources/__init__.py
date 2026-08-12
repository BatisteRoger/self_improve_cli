"""Trace source protocol and adapters.

A TraceSource is anything that can list and fetch traces. The initial
implementation is a LangSmith adapter. Provider-specific SDK objects stop
at this boundary — everything downstream works on the canonical domain model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from self_improve_cli.domain import Run as Run
from self_improve_cli.domain import RunSummary, Trace


class TraceSource(ABC):
    """Protocol for trace providers (LangSmith, other backends, local files)."""

    @abstractmethod
    def list_root_runs(self, limit: int = 20) -> list[RunSummary]:
        """List recent root runs (one per trace), most recent first."""
        ...

    @abstractmethod
    def list_runs_by_type(self, run_type: str, limit: int = 20) -> list[RunSummary]:
        """List recent runs of a given type across all traces."""
        ...

    @abstractmethod
    def fetch_trace(self, trace_id: str) -> Trace:
        """Download all runs of a trace and return a canonical Trace."""
        ...

    @abstractmethod
    def resolve_trace_id(self, run_id: str) -> str:
        """Resolve a run ID to its parent trace ID.

        Useful when the user only has a run ID (e.g. from a LangSmith URL)
        and needs the trace ID to fetch the full trace.
        """
        ...


def _content_text(content: Any) -> str:
    """Extract plain text from a message content (string or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _normalize_message(m: dict[str, Any]) -> dict[str, Any]:
    """Normalize a serialized message (LC constructor format or plain dict)."""
    if "kwargs" in m and "id" in m and isinstance(m["id"], list):
        role = str(m["id"][-1]).removesuffix("Message").lower()
        kw = m["kwargs"]
    else:
        role = m.get("type") or m.get("role") or "?"
        kw = m
    return {
        "role": role,
        "text": _content_text(kw.get("content")),
        "tool_calls": kw.get("tool_calls") or [],
        "tool_call_id": kw.get("tool_call_id"),
    }


def _input_messages(run_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized input messages from an LLM run dict."""
    msgs = (run_dict.get("inputs") or {}).get("messages") or []
    if len(msgs) == 1 and isinstance(msgs[0], list):
        msgs = msgs[0]
    return [_normalize_message(m) for m in msgs if isinstance(m, dict)]


def _output_message(run_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Extract normalized output message from an LLM run dict."""
    outputs = run_dict.get("outputs") or {}
    generations = outputs.get("generations")
    if generations:
        gen = generations[0]
        if isinstance(gen, list) and gen:
            gen = gen[0]
        if isinstance(gen, dict) and isinstance(gen.get("message"), dict):
            return _normalize_message(gen["message"])
    return None
