"""Tests for the LangSmith source adapter (SmithDB-backed methods).

These tests mock the SDK client to verify the adapter calls the new
SmithDB-backed methods with correct parameters. No network access.
"""

from __future__ import annotations

from typing import Any

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
    def __init__(self, id: str = "proj-uuid-123"):
        self.id = id


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
        read_run_result: Any | None = None,
    ) -> None:
        self.traces = _MockTraces(traces=traces, runs=runs)
        self.runs = _MockRuns(
            runs=runs,
            retrieve_result=retrieve_result,
            retrieve_exc=retrieve_exc,
        )
        self._project = project or _MockProject()
        self._read_run_result = read_run_result
        self.aread_project_calls: list[dict] = []
        self.read_run_calls: list[dict] = []

    async def aread_project(self, project_name: str | None = None, **_kw: Any) -> _MockProject:
        self.aread_project_calls.append({"project_name": project_name})
        return self._project

    def read_run(self, run_id: str) -> Any:
        self.read_run_calls.append({"run_id": run_id})
        return self._read_run_result


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
# Tests: resolve_trace_id (SmithDB-native + legacy fallback)
# ---------------------------------------------------------------------------


def test_resolve_trace_id_smithdb_native(monkeypatch):
    """When a project is configured and the run is in it, uses runs.retrieve."""
    from langsmith import NotFoundError  # noqa: F401 — ensure import works

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
    # Legacy read_run was NOT called
    assert client.read_run_calls == []


def test_resolve_trace_id_falls_back_on_not_found(monkeypatch):
    """When the run is not in the configured project (404), falls back to read_run."""
    import httpx
    from langsmith import NotFoundError

    _404 = httpx.Response(404, request=httpx.Request("GET", "https://example.com"))
    legacy_run = _MockRun(
        id="run-abc",
        trace_id="trace-from-legacy",
        session_id="other-proj-uuid",
    )
    client = _MockClient(
        retrieve_exc=NotFoundError("not found", response=_404, body=None),
        read_run_result=legacy_run,
    )

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)
    resolution = source.resolve_trace_id("run-abc")

    # Fell back to read_run
    assert resolution.trace_id == "trace-from-legacy"
    assert resolution.project_id == "other-proj-uuid"
    assert len(client.runs.retrieve_calls) == 1
    assert len(client.read_run_calls) == 1


def test_resolve_trace_id_no_project_uses_legacy(monkeypatch):
    """When no project is configured, uses legacy read_run directly."""
    legacy_run = _MockRun(
        id="run-abc",
        trace_id="trace-legacy",
        session_id="some-proj-uuid",
    )
    client = _MockClient(read_run_result=legacy_run)

    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    source = LangSmithSource(client=client)
    resolution = source.resolve_trace_id("run-abc")

    assert resolution.trace_id == "trace-legacy"
    # runs.retrieve was NOT attempted (no project to try)
    assert client.runs.retrieve_calls == []
    assert len(client.read_run_calls) == 1


def test_resolve_trace_id_non_404_error_propagates(monkeypatch):
    """Non-404 errors from runs.retrieve propagate (no legacy fallback)."""

    # APIStatusError requires a response object; use a plain RuntimeError
    # to verify non-NotFoundError exceptions are not swallowed.
    client = _MockClient(retrieve_exc=RuntimeError("network error"))

    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    source = LangSmithSource(client=client)

    try:
        source.resolve_trace_id("run-abc")
        raise AssertionError("Expected RuntimeError to propagate")
    except RuntimeError:
        pass  # Expected — non-404 errors should not trigger fallback

    # read_run was NOT called (error propagated before fallback)
    assert client.read_run_calls == []
