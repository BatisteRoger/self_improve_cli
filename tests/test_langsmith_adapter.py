"""Tests for the LangSmith source adapter (SmithDB-backed methods).

These tests mock the SDK client to verify the adapter calls the new
SmithDB-backed methods with correct parameters. No network access.
"""

from __future__ import annotations

from typing import Any

import pytest

from self_improve_cli.domain import RunType
from self_improve_cli.sources.langsmith import (
    LangSmithSource,
    _parse_run_type,
    _sdk_run_to_canonical,
    _trace_to_summary,
)

# ---------------------------------------------------------------------------
# Mock SDK objects
# ---------------------------------------------------------------------------


class _MockProject:
    def __init__(self, id: str = "proj-uuid-123", name: str = "test-project"):
        self.id = id
        self.name = name


class _MockRun:
    """Mock SDK Run object (pydantic-like)."""

    def __init__(self, session_id: str | None = None, **kwargs: Any) -> None:
        self.session_id = session_id
        for k, v in kwargs.items():
            setattr(self, k, v)

    def dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class _MockTraceAggregates:
    def __init__(self, total_tokens: int | None = None, total_cost: float | None = None):
        self.total_tokens = total_tokens
        self.total_cost = total_cost


class _MockTrace:
    """Mock SDK Trace object from traces.query."""

    def __init__(self, root_run: _MockRun, trace_aggregates: _MockTraceAggregates | None = None):
        self.root_run = root_run
        self.trace_aggregates = trace_aggregates


class _MockTraceListRunsResponse:
    def __init__(self, items: list[Any] | None = None):
        self.items = items


class _AsyncPaginatorMock:
    """Supports `async for` — mimics the SDK's AsyncPaginator."""

    def __init__(self, items: list[Any]):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        if self._items:
            return self._items.pop(0)
        raise StopAsyncIteration


class _MockRuns:
    def __init__(
        self,
        runs: list[Any] | None = None,
        retrieve_result: Any | None = None,
        retrieve_exc: Exception | None = None,
    ) -> None:
        self._runs = runs or []
        self.query_calls: list[dict] = []
        self.retrieve_calls: list[dict] = []
        self._retrieve_result = retrieve_result
        self._retrieve_exc = retrieve_exc

    def query(self, **kwargs: Any) -> _AsyncPaginatorMock:
        self.query_calls.append(kwargs)
        return _AsyncPaginatorMock(list(self._runs))

    async def retrieve(self, run_id: str, *, project_id: str, **kwargs: Any) -> Any:
        self.retrieve_calls.append({"run_id": run_id, "project_id": project_id, **kwargs})
        if self._retrieve_exc is not None:
            raise self._retrieve_exc
        return self._retrieve_result


class _MockTraces:
    def __init__(self, traces: list[Any] | None = None, runs: list[Any] | None = None):
        self._traces = traces or []
        self._runs = runs or []
        self.list_runs_calls: list[dict] = []

    def query(self, **kwargs: Any) -> _AsyncPaginatorMock:
        return _AsyncPaginatorMock(list(self._traces))

    async def list_runs(self, trace_id: str, *, project_id: str, **kwargs: Any) -> Any:
        self.list_runs_calls.append({"trace_id": trace_id, "project_id": project_id, **kwargs})
        return _MockTraceListRunsResponse(list(self._runs))


class _MockClient:
    def __init__(
        self,
        traces: list[Any] | None = None,
        runs: list[Any] | None = None,
        project: _MockProject | None = None,
        retrieve_result: Any | None = None,
        retrieve_exc: Exception | None = None,
    ) -> None:
        self.traces = _MockTraces(traces=traces, runs=runs)
        self.runs = _MockRuns(
            runs=runs,
            retrieve_result=retrieve_result,
            retrieve_exc=retrieve_exc,
        )
        self._project = project or _MockProject()
        self.aread_project_calls: list[dict] = []

    async def aread_project(self, project_name: str | None = None, **kw: Any) -> _MockProject:
        call: dict[str, Any] = {"project_name": project_name}
        if "project_id" in kw:
            call["project_id"] = kw["project_id"]
        self.aread_project_calls.append(call)
        return self._project


# ---------------------------------------------------------------------------
# Helpers to build synthetic runs
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str = "run-1",
    trace_id: str = "trace-1",
    run_type: str = "LLM",
    name: str = "test-run",
    parent_run_ids: list[str] | None = None,
    dotted_order: str = "20240101T000000.000000Z",
    status: str = "SUCCESS",
    start_time: Any = None,
    end_time: Any = None,
    total_tokens: int | None = 100,
    prompt_tokens: int | None = 50,
    completion_tokens: int | None = 50,
    error: str | None = None,
    inputs: dict | None = None,
    outputs: dict | None = None,
) -> _MockRun:
    return _MockRun(
        id=run_id,
        trace_id=trace_id,
        run_type=run_type,
        name=name,
        parent_run_ids=parent_run_ids,
        dotted_order=dotted_order,
        status=status,
        start_time=start_time,
        end_time=end_time,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error=error,
        inputs=inputs or {},
        outputs=outputs or {},
    )


# ---------------------------------------------------------------------------
# Tests: _parse_run_type
# ---------------------------------------------------------------------------


def test_parse_run_type_uppercase():
    """SmithDB API returns uppercase run_type values; our enum is lowercase."""
    assert _parse_run_type("LLM") == RunType.LLM
    assert _parse_run_type("TOOL") == RunType.TOOL
    assert _parse_run_type("CHAIN") == RunType.CHAIN


def test_parse_run_type_lowercase_still_works():
    """Legacy API returned lowercase; still supported for backward compat."""
    assert _parse_run_type("llm") == RunType.LLM
    assert _parse_run_type("tool") == RunType.TOOL


def test_parse_run_type_unknown_falls_to_other():
    assert _parse_run_type("RETRIEVER") == RunType.OTHER
    assert _parse_run_type("EMBEDDING") == RunType.OTHER
    assert _parse_run_type(None) == RunType.OTHER


# ---------------------------------------------------------------------------
# Tests: _sdk_run_to_canonical with parent_run_ids
# ---------------------------------------------------------------------------


def test_sdk_run_to_canonical_parent_run_ids_last_is_direct_parent():
    """SmithDB returns parent_run_ids (root first); direct parent is last."""
    run = _make_run(parent_run_ids=["root-id", "chain-id", "llm-parent-id"])
    canonical = _sdk_run_to_canonical(run)
    assert canonical.parent_run_id == "llm-parent-id"


def test_sdk_run_to_canonical_empty_parent_run_ids():
    """Root run has empty parent_run_ids list."""
    run = _make_run(parent_run_ids=[])
    canonical = _sdk_run_to_canonical(run)
    assert canonical.parent_run_id is None


def test_sdk_run_to_canonical_single_parent_run_id():
    """One ancestor in the list — that's the direct parent."""
    run = _make_run(parent_run_ids=["only-parent"])
    canonical = _sdk_run_to_canonical(run)
    assert canonical.parent_run_id == "only-parent"


# ---------------------------------------------------------------------------
# Tests: _trace_to_summary
# ---------------------------------------------------------------------------


def test_trace_to_summary_reads_total_tokens_from_aggregates():
    """traces.query moves total_tokens to trace_aggregates, not root_run."""
    root = _make_run(total_tokens=999)  # root_run's own tokens (not the trace total)
    agg = _MockTraceAggregates(total_tokens=5000)
    trace = _MockTrace(root_run=root, trace_aggregates=agg)
    summary = _trace_to_summary(trace)
    assert summary.total_tokens == 5000


def test_trace_to_summary_no_aggregates():
    """When trace_aggregates is None, total_tokens is None."""
    root = _make_run(total_tokens=999)
    trace = _MockTrace(root_run=root, trace_aggregates=None)
    summary = _trace_to_summary(trace)
    assert summary.total_tokens is None


def test_trace_to_summary_basic_fields():
    root = _make_run(run_id="r1", trace_id="t1", name="my-run", run_type="LLM", status="SUCCESS")
    trace = _MockTrace(root_run=root)
    summary = _trace_to_summary(trace)
    assert summary.id == "r1"
    assert summary.trace_id == "t1"
    assert summary.name == "my-run"
    assert summary.run_type == RunType.LLM
    assert summary.status == "SUCCESS"


# ---------------------------------------------------------------------------
# Tests: list_root_runs (traces.query)
# ---------------------------------------------------------------------------


def test_list_root_runs_calls_traces_query(monkeypatch):
    """list_root_runs uses traces.query with project_id and selects."""
    root1 = _make_run(run_id="r1", trace_id="t1")
    root2 = _make_run(run_id="r2", trace_id="t2")
    trace1 = _MockTrace(root_run=root1)
    trace2 = _MockTrace(root_run=root2)
    client = _MockClient(traces=[trace1, trace2])

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)
    summaries = source.list_root_runs(limit=10)

    assert len(summaries) == 2
    assert summaries[0].id == "r1"
    assert summaries[1].id == "r2"
    # Project UUID was resolved
    assert client.aread_project_calls == [{"project_name": "test-project"}]


def test_list_root_runs_respects_limit(monkeypatch):
    """list_root_runs stops after reaching the limit."""
    traces = [_MockTrace(root_run=_make_run(run_id=f"r{i}")) for i in range(50)]
    client = _MockClient(traces=traces)

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)
    summaries = source.list_root_runs(limit=5)

    assert len(summaries) == 5


# ---------------------------------------------------------------------------
# Tests: list_runs_by_type (runs.query)
# ---------------------------------------------------------------------------


def test_list_runs_by_type_calls_runs_query(monkeypatch):
    """list_runs_by_type uses runs.query with project_ids and uppercase run_type."""
    run1 = _make_run(run_id="r1", trace_id="t1", run_type="LLM")
    run2 = _make_run(run_id="r2", trace_id="t2", run_type="LLM")
    client = _MockClient(runs=[run1, run2])

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)
    summaries = source.list_runs_by_type("llm", limit=10)

    assert len(summaries) == 2
    assert client.runs.query_calls[0]["run_type"] == "LLM"
    assert client.runs.query_calls[0]["project_ids"] == ["proj-uuid-123"]


# ---------------------------------------------------------------------------
# Tests: fetch_trace (traces.list_runs)
# ---------------------------------------------------------------------------


def test_fetch_trace_calls_traces_list_runs(monkeypatch):
    """fetch_trace uses traces.list_runs with trace_id, project_id, and full selects."""
    run1 = _make_run(run_id="r1", trace_id="t1", parent_run_ids=[])
    run2 = _make_run(
        run_id="r2", trace_id="t1", parent_run_ids=["r1"], dotted_order="20240101T000001.000000Z"
    )
    client = _MockClient(runs=[run1, run2])

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)
    trace = source.fetch_trace("t1")

    assert trace.trace_id == "t1"
    assert len(trace.runs) == 2
    # Sorted by dotted_order
    assert trace.runs[0].id == "r1"
    assert trace.runs[1].id == "r2"
    # traces.list_runs was called with correct params
    call = client.traces.list_runs_calls[0]
    assert call["trace_id"] == "t1"
    assert call["project_id"] == "proj-uuid-123"
    assert "PARENT_RUN_IDS" in call["selects"]
    assert "INPUTS" in call["selects"]


def test_fetch_trace_with_explicit_project_id(monkeypatch):
    """When project_id is provided, fetch_trace uses it directly (no resolution)."""
    run1 = _make_run(run_id="r1", trace_id="t1")
    client = _MockClient(runs=[run1])

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)
    trace = source.fetch_trace("t1", project_id="other-proj-uuid")

    assert trace.trace_id == "t1"
    assert client.traces.list_runs_calls[0]["project_id"] == "other-proj-uuid"
    # aread_project should NOT have been called
    assert client.aread_project_calls == []


def test_fetch_trace_empty(monkeypatch):
    """fetch_trace returns empty trace when no runs found."""
    client = _MockClient(runs=[])

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)
    trace = source.fetch_trace("nonexistent")

    assert trace.trace_id == "nonexistent"
    assert len(trace.runs) == 0


# ---------------------------------------------------------------------------
# Tests: project_id caching
# ---------------------------------------------------------------------------


def test_project_id_resolution_caches(monkeypatch):
    """aread_project is called once; subsequent calls use the cached UUID."""
    root = _make_run(run_id="r1")
    trace = _MockTrace(root_run=root)
    client = _MockClient(traces=[trace])

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    source.list_root_runs(limit=5)
    source.list_root_runs(limit=5)

    assert len(client.aread_project_calls) == 1


# ---------------------------------------------------------------------------
# Tests: resolve_trace_id (runs.retrieve only, no global lookup)
# ---------------------------------------------------------------------------
# The mock client has no read_run: the deprecated global lookup (removed by
# LangSmith on 31 Jan 2027) would fail these tests with AttributeError.


def test_resolve_trace_id_smithdb_native(monkeypatch):
    """When a project is configured and the run is in it, uses runs.retrieve."""
    retrieve_result = _MockRun(
        id="run-abc",
        trace_id="trace-resolved",
        project_id="proj-uuid-123",
    )
    client = _MockClient(retrieve_result=retrieve_result)

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)
    resolution = source.resolve_trace_id("run-abc")

    assert resolution.trace_id == "trace-resolved"
    assert resolution.project_id == "proj-uuid-123"
    # runs.retrieve was called with the configured project_id
    assert client.runs.retrieve_calls[0]["run_id"] == "run-abc"
    assert client.runs.retrieve_calls[0]["project_id"] == "proj-uuid-123"
    assert not hasattr(client, "read_run")


def test_resolve_trace_id_not_found_raises_clear_error(monkeypatch):
    """A run missing from the configured project raises a RuntimeError naming the project."""
    import httpx
    from langsmith import NotFoundError

    _404 = httpx.Response(404, request=httpx.Request("GET", "https://example.com"))
    client = _MockClient(retrieve_exc=NotFoundError("not found", response=_404, body=None))

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    with pytest.raises(RuntimeError, match=r"not found in project 'test-project'.*--project"):
        source.resolve_trace_id("run-abc")

    assert len(client.runs.retrieve_calls) == 1


def test_resolve_trace_id_no_project_raises_clear_error(monkeypatch):
    """Without a project, raises a RuntimeError asking for LANGSMITH_PROJECT or --project."""
    client = _MockClient()

    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    source = LangSmithSource(client=client)

    with pytest.raises(RuntimeError, match=r"LANGSMITH_PROJECT.*--project"):
        source.resolve_trace_id("run-abc")

    # No API call at all: neither project lookup nor runs.retrieve
    assert client.aread_project_calls == []
    assert client.runs.retrieve_calls == []


def test_resolve_trace_id_non_404_error_propagates(monkeypatch):
    """Non-404 errors from runs.retrieve propagate unchanged."""
    client = _MockClient(retrieve_exc=RuntimeError("network error"))

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    with pytest.raises(RuntimeError, match="network error"):
        source.resolve_trace_id("run-abc")


# ---------------------------------------------------------------------------
# Tests: resolve_trace_id from a LangSmith run URL / UUIDv7 start_time
# ---------------------------------------------------------------------------

_ORG = "11111111-1111-4111-8111-111111111111"
_PROJ = "22222222-2222-4222-8222-222222222222"
_TRACE = "019ff0e5-0000-7000-8000-000000000001"
_RUN7 = "019ff0e5-7189-713d-8ab9-c032edf9d4dd"


def _not_found() -> Exception:
    import httpx
    from langsmith import NotFoundError

    resp = httpx.Response(404, request=httpx.Request("GET", "https://example.com"))
    return NotFoundError("not found", response=resp, body=None)


def test_resolve_from_complete_url_makes_no_api_call(monkeypatch):
    """A URL with project and trace IDs resolves with no API call."""
    client = _MockClient()
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    source = LangSmithSource(client=client)

    url = f"https://smith.example.com/o/{_ORG}/projects/p/{_PROJ}/trace/{_TRACE}/run/{_RUN7}"
    resolution = source.resolve_trace_id(url)

    assert resolution.trace_id == _TRACE
    assert resolution.project_id == _PROJ
    assert client.runs.retrieve_calls == []
    assert client.aread_project_calls == []


def test_resolve_from_url_without_trace_retrieves_in_url_project(monkeypatch):
    """A URL without trace ID calls runs.retrieve in the URL's project, not the configured one."""
    retrieve_result = _MockRun(id=_RUN7, trace_id=_TRACE, project_id=_PROJ)
    client = _MockClient(retrieve_result=retrieve_result)
    monkeypatch.setenv("LANGSMITH_PROJECT", "other-project")
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    source = LangSmithSource(client=client)

    url = f"https://smith.example.com/o/{_ORG}/projects/p/{_PROJ}/r/{_RUN7}"
    resolution = source.resolve_trace_id(url)

    assert resolution.trace_id == _TRACE
    call = client.runs.retrieve_calls[0]
    assert call["run_id"] == _RUN7
    assert call["project_id"] == _PROJ
    # No lookup of the configured project
    assert client.aread_project_calls == []


def test_resolve_from_url_enforces_allowlist(monkeypatch):
    """With an allowlist, the URL's project name is read from its ID and checked."""
    client = _MockClient(project=_MockProject(id=_PROJ, name="production"))
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "my-agent-*")
    source = LangSmithSource(client=client)

    url = f"https://smith.example.com/o/{_ORG}/projects/p/{_PROJ}/trace/{_TRACE}/run/{_RUN7}"
    with pytest.raises(PermissionError, match="production"):
        source.resolve_trace_id(url)

    assert client.aread_project_calls == [{"project_name": None, "project_id": _PROJ}]
    assert client.runs.retrieve_calls == []


def test_resolve_from_url_allowed_project_passes(monkeypatch):
    """An allowlisted URL project resolves after the name check."""
    client = _MockClient(project=_MockProject(id=_PROJ, name="my-agent-dev"))
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "my-agent-*")
    source = LangSmithSource(client=client)

    url = f"https://smith.example.com/o/{_ORG}/projects/p/{_PROJ}/trace/{_TRACE}/run/{_RUN7}"
    assert source.resolve_trace_id(url).trace_id == _TRACE


def test_resolve_from_malformed_url_raises_runtime_error(monkeypatch):
    """A URL that is not a LangSmith run URL fails cleanly, with no API call."""
    client = _MockClient()
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    with pytest.raises(RuntimeError, match="Not a LangSmith run URL"):
        source.resolve_trace_id("https://example.com/some/page")
    assert client.runs.retrieve_calls == []


def test_resolve_uuid7_run_id_passes_start_time(monkeypatch):
    """A UUIDv7 run ID passes its embedded timestamp as start_time."""
    from datetime import UTC, datetime

    retrieve_result = _MockRun(id=_RUN7, trace_id=_TRACE, project_id="proj-uuid-123")
    client = _MockClient(retrieve_result=retrieve_result)
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    source.resolve_trace_id(_RUN7)

    start_time = client.runs.retrieve_calls[0]["start_time"]
    assert isinstance(start_time, datetime)
    assert start_time.tzinfo == UTC
    assert start_time.year >= 2025


def test_resolve_url_start_time_wins_over_uuid7(monkeypatch):
    """The URL's start_time is passed as is when present."""
    from datetime import UTC, datetime

    retrieve_result = _MockRun(id=_RUN7, trace_id=_TRACE, project_id=_PROJ)
    client = _MockClient(retrieve_result=retrieve_result)
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    source = LangSmithSource(client=client)

    url = (
        f"https://smith.example.com/o/{_ORG}/projects/p/{_PROJ}/r/{_RUN7}"
        "?start_time=2026-11-06T16:36:51.661014Z"
    )
    source.resolve_trace_id(url)

    assert client.runs.retrieve_calls[0]["start_time"] == datetime(
        2026, 11, 6, 16, 36, 51, 661014, tzinfo=UTC
    )


def test_resolve_non_uuid7_run_id_omits_start_time(monkeypatch):
    """A non-v7 run ID sends no start_time."""
    retrieve_result = _MockRun(id="r", trace_id=_TRACE, project_id="proj-uuid-123")
    client = _MockClient(retrieve_result=retrieve_result)
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    source.resolve_trace_id("6f1c2b1e-8a4d-4c1e-9f1a-2b3c4d5e6f70")

    assert "start_time" not in client.runs.retrieve_calls[0]


def test_resolve_uuid7_not_found_is_a_clean_error_without_retry(monkeypatch):
    """A 404 with a start_time is reported once: no retry without start_time
    (the live API answers that retry with HTTP 500 for a run in another project)."""
    client = _MockClient(retrieve_exc=_not_found())
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    with pytest.raises(RuntimeError, match=r"not found in project 'test-project'"):
        source.resolve_trace_id(_RUN7)
    assert len(client.runs.retrieve_calls) == 1
    assert "start_time" in client.runs.retrieve_calls[0]


def test_resolve_not_found_suggests_url(monkeypatch):
    """The not-found error suggests pasting the run URL or --project."""
    client = _MockClient(retrieve_exc=_not_found())
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    with pytest.raises(RuntimeError, match=r"run URL.*--project"):
        source.resolve_trace_id("run-abc")
