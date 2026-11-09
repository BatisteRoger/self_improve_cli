"""Parse LangSmith UI run URLs and UUIDv7 run IDs into lookup hints.

Pure functions, no network. Only IDs and the start time are extracted: the
URL host is never used for API calls (those go to the configured endpoint).

Accepted URL shapes (any host, so self-hosted deployments work):

- ``/o/<org>/projects/p/<project_id>/trace/<trace_id>/run/<run_id>?start_time=<iso>``
  (what ``runs.get_url`` returns; ``/run/<run_id>`` may be absent for a root run)
- ``/o/<org>/projects/p/<project_id>/r/<run_id>?trace_id=<trace_id>&start_time=<iso>``
- ``/o/<org>/projects/p/<project_id>?peek=<run_id>&peeked_trace=<trace_id>``
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class LangSmithRunRef:
    """IDs extracted from a LangSmith UI run URL. Missing parts are None."""

    run_id: str
    project_id: str | None = None
    trace_id: str | None = None
    start_time: datetime | None = None


def is_url(text: str) -> bool:
    """True when the text looks like an http(s) URL rather than a bare ID."""
    return text.strip().lower().startswith(("http://", "https://"))


def _uuid_or_none(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _segment_after(segments: list[str], key: str) -> str | None:
    """Return the path segment following ``key``, if any."""
    for i, seg in enumerate(segments[:-1]):
        if seg == key:
            return segments[i + 1]
    return None


def parse_langsmith_run_url(url: str) -> LangSmithRunRef:
    """Extract run, project, and trace IDs plus start time from a LangSmith UI URL.

    Raises ValueError when the URL is not a LangSmith project/run URL or
    carries no run ID.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Not an http(s) URL: {url!r}")
    segments = [s for s in parts.path.split("/") if s]
    query = {k: v[0] for k, v in parse_qs(parts.query).items() if v}

    project_id = None
    for i in range(len(segments) - 2):
        if segments[i] == "projects" and segments[i + 1] == "p":
            project_id = _uuid_or_none(segments[i + 2])
            break
    if project_id is None:
        raise ValueError(
            "Not a LangSmith run URL (expected .../projects/p/<project_id>/...). "
            "Paste the URL of the run page from the LangSmith UI."
        )

    trace_id = (
        _uuid_or_none(_segment_after(segments, "trace"))
        or _uuid_or_none(query.get("trace_id"))
        or _uuid_or_none(query.get("peeked_trace"))
    )
    run_id = (
        _uuid_or_none(_segment_after(segments, "run"))
        or _uuid_or_none(_segment_after(segments, "r"))
        or _uuid_or_none(query.get("peek"))
        or trace_id  # a trace URL with no run selected points at the root run
    )
    if run_id is None:
        raise ValueError(
            "LangSmith URL has no run ID. Open the run in the LangSmith UI and copy "
            "the URL of the run page."
        )
    return LangSmithRunRef(
        run_id=run_id,
        project_id=project_id,
        trace_id=trace_id,
        start_time=_parse_time(query.get("start_time")),
    )


def uuid7_timestamp(run_id: str) -> datetime | None:
    """Return the millisecond timestamp embedded in a UUIDv7, else None."""
    try:
        parsed = uuid.UUID(run_id)
    except ValueError:
        return None
    if parsed.version != 7:
        return None
    millis = int.from_bytes(parsed.bytes[:6], "big")
    return datetime.fromtimestamp(millis / 1000, tz=UTC)
