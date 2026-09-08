"""LangSmith trace source adapter.

Normalizes LangSmith SDK Run objects into the canonical domain model.
SDK-specific objects stop here — nothing downstream imports langsmith.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from self_improve_cli.domain import (
    Message,
    ProjectSummary,
    Run,
    RunResolution,
    RunSummary,
    RunType,
    ToolCall,
    Trace,
)
from self_improve_cli.sources import (
    TraceSource,
    _input_messages,
    _output_message,
)

logger = logging.getLogger(__name__)

# SmithDB-backed API uses uppercase SCREAMING_SNAKE_CASE selects.
# Default selects is ["ID"] only — we must explicitly request every field.
_LIST_SELECTS = [
    "ID",
    "TRACE_ID",
    "NAME",
    "RUN_TYPE",
    "STATUS",
    "START_TIME",
    "END_TIME",
    "ERROR",
    "TOTAL_TOKENS",
    "PROMPT_TOKENS",
    "COMPLETION_TOKENS",
]

# Full field set for fetch_trace — everything _sdk_run_to_canonical reads.
_FULL_SELECTS = [
    "ID",
    "TRACE_ID",
    "RUN_TYPE",
    "NAME",
    "PARENT_RUN_IDS",
    "DOTTED_ORDER",
    "STATUS",
    "START_TIME",
    "END_TIME",
    "TOTAL_TOKENS",
    "PROMPT_TOKENS",
    "COMPLETION_TOKENS",
    "ERROR",
    "INPUTS",
    "OUTPUTS",
]

# traces.query defaults min_start_time to 24h ago, which would silently
# hide older traces. Use a wide window to preserve the legacy "most recent
# N regardless of age" behaviour.
_LIST_MIN_START_DAYS = 90


def _get_client(workspace_id: str | None = None) -> Any:
    """Build a LangSmith client from environment variables.

    Requires LANGSMITH_API_KEY in the environment. Raises RuntimeError if
    missing.

    Args:
        workspace_id: Optional LangSmith workspace UUID for non-default
            workspaces (e.g. Flows). If None, uses the default workspace.
    """
    try:
        from langsmith import Client  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "langsmith is not installed. Install with: pip install 'self-improve-cli[langsmith]'"
        ) from exc

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY is not set. Run `self-improve init` to create a .env file."
        )
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    logger.debug(
        "Creating LangSmith client (endpoint=%s, workspace=%s)", endpoint, workspace_id or "default"
    )
    kwargs: dict[str, Any] = {"api_url": endpoint, "api_key": api_key}
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    return Client(**kwargs)


def _get_project_name() -> str:
    """Return the LangSmith project name from the environment."""
    return os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or ""


def _get_allowed_projects() -> list[str]:
    """Return the list of allowed project patterns from the environment.

    Empty/unset means all projects are allowed (gated by API key permissions).
    """
    raw = os.environ.get("LANGSMITH_ALLOWED_PROJECTS", "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _check_project_allowed(project: str) -> None:
    """Raise PermissionError if the project is not in the allowlist.

    If no allowlist is configured, all projects are allowed.
    Patterns use fnmatch syntax (*, ?, [seq]).
    """
    allowed = _get_allowed_projects()
    if not allowed:
        return
    if not any(fnmatch.fnmatch(project, pattern) for pattern in allowed):
        raise PermissionError(
            f"Project '{project}' is not in the allowed list. "
            f"Allowed patterns: {', '.join(allowed)}. "
            "Set LANGSMITH_ALLOWED_PROJECTS in .env to adjust, or leave it empty "
            "to allow all projects."
        )


def _parse_run_type(raw: str | None) -> RunType:
    if raw is None:
        return RunType.OTHER
    try:
        # SmithDB API returns uppercase ("LLM", "TOOL", ...); legacy API
        # returned lowercase. Our enum values are lowercase, so normalise.
        return RunType(raw.lower() if isinstance(raw, str) else raw)
    except ValueError:
        return RunType.OTHER


def _sdk_run_to_summary(sdk_run: Any) -> RunSummary:
    """Convert a LangSmith SDK Run (lightweight) to a RunSummary."""
    return RunSummary(
        id=str(sdk_run.id),
        trace_id=str(sdk_run.trace_id),
        name=sdk_run.name,
        run_type=_parse_run_type(getattr(sdk_run, "run_type", None)),
        status=sdk_run.status,
        start_time=str(sdk_run.start_time) if sdk_run.start_time else None,
        total_tokens=sdk_run.total_tokens,
        error=sdk_run.error,
    )


def _trace_to_summary(trace: Any) -> RunSummary:
    """Convert a SmithDB Trace (from traces.query) to a RunSummary.

    traces.query returns Trace objects with root_run + trace_aggregates.
    total_tokens lives on trace_aggregates, not on root_run.
    """
    root = trace.root_run
    if root is None:
        raise ValueError("Trace has no root_run")
    agg = getattr(trace, "trace_aggregates", None)
    total_tokens = getattr(agg, "total_tokens", None) if agg else None
    return RunSummary(
        id=str(root.id),
        trace_id=str(root.trace_id),
        name=root.name,
        run_type=_parse_run_type(getattr(root, "run_type", None)),
        status=root.status,
        start_time=str(root.start_time) if root.start_time else None,
        total_tokens=total_tokens,
        error=root.error,
    )


def _run_async(coro: Any) -> Any:
    """Run a coroutine from synchronous context.

    The CLI is fully synchronous. The SmithDB-backed SDK methods are async,
    so we wrap each call in asyncio.run(). This is safe because the CLI
    never runs inside an existing event loop.
    """
    return asyncio.run(coro)


def _sdk_run_to_canonical(sdk_run: Any) -> Run:
    """Convert a full LangSmith SDK Run to a canonical Run."""
    run_dict = sdk_run.dict() if hasattr(sdk_run, "dict") else dict(sdk_run)

    input_msgs_raw = _input_messages(run_dict)
    output_msg_raw = _output_message(run_dict)

    input_messages = [
        Message(
            role=m["role"],
            text=m["text"],
            tool_calls=[
                ToolCall(
                    name=tc.get("name", "?"),
                    args=tc.get("args", {}) if isinstance(tc.get("args"), dict) else {},
                    id=tc.get("id"),
                )
                for tc in m["tool_calls"]
                if isinstance(tc, dict)
            ],
            tool_call_id=m["tool_call_id"],
        )
        for m in input_msgs_raw
    ]

    output_message = None
    if output_msg_raw:
        output_message = Message(
            role=output_msg_raw["role"],
            text=output_msg_raw["text"],
            tool_calls=[
                ToolCall(
                    name=tc.get("name", "?"),
                    args=tc.get("args", {}) if isinstance(tc.get("args"), dict) else {},
                    id=tc.get("id"),
                )
                for tc in output_msg_raw["tool_calls"]
                if isinstance(tc, dict)
            ],
            tool_call_id=output_msg_raw["tool_call_id"],
        )

    # SmithDB API replaces parent_run_id (single) with parent_run_ids
    # (list of all ancestors, root first). The direct parent is the last
    # element. Fall back to parent_run_id for legacy API compatibility.
    _parent_ids = run_dict.get("parent_run_ids") or []
    _pid = _parent_ids[-1] if _parent_ids else run_dict.get("parent_run_id")
    _dotted = run_dict.get("dotted_order")
    return Run(
        id=str(run_dict.get("id", "")),
        trace_id=str(run_dict.get("trace_id", "")),
        run_type=_parse_run_type(run_dict.get("run_type")),
        name=run_dict.get("name", ""),
        parent_run_id=str(_pid) if _pid is not None else None,
        dotted_order=str(_dotted) if _dotted is not None else None,
        status=run_dict.get("status"),
        start_time=str(run_dict.get("start_time")) if run_dict.get("start_time") else None,
        end_time=str(run_dict.get("end_time")) if run_dict.get("end_time") else None,
        total_tokens=run_dict.get("total_tokens"),
        prompt_tokens=run_dict.get("prompt_tokens"),
        completion_tokens=run_dict.get("completion_tokens"),
        error=run_dict.get("error"),
        inputs=run_dict.get("inputs") or {},
        outputs=run_dict.get("outputs") or {},
        input_messages=input_messages,
        output_message=output_message,
        extra={
            k: v
            for k, v in run_dict.items()
            if k
            not in {
                "id",
                "trace_id",
                "run_type",
                "name",
                "parent_run_id",
                "parent_run_ids",
                "dotted_order",
                "status",
                "start_time",
                "end_time",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "error",
                "inputs",
                "outputs",
            }
        },
    )


class LangSmithSource(TraceSource):
    """LangSmith trace source. Requires LANGSMITH_API_KEY in the environment."""

    def __init__(self, project_name: str | None = None, client: Any | None = None) -> None:
        self._project_name = project_name or _get_project_name()
        self._client = client
        self._project_id_cache: str | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _get_client()
        return self._client

    def _resolve_project(self) -> str:
        if not self._project_name:
            raise RuntimeError(
                "No LangSmith project name configured. Set LANGSMITH_PROJECT in .env "
                "or pass --project."
            )
        _check_project_allowed(self._project_name)
        return self._project_name

    async def _resolve_project_id_async(self) -> str:
        """Resolve the configured project name to a SmithDB project UUID.

        SmithDB-backed methods require project_ids (UUIDs), not project
        names. Caches the result on the instance to avoid repeated lookups.
        """
        if self._project_id_cache is not None:
            return self._project_id_cache
        project_name = self._resolve_project()
        client = self._get_client()
        logger.info("Resolving project UUID for %s", project_name)
        project = await client.aread_project(project_name=project_name)
        self._project_id_cache = str(project.id)
        logger.info("Project %s -> UUID %s", project_name, self._project_id_cache)
        return self._project_id_cache

    def _resolve_project_id(self) -> str:
        return _run_async(self._resolve_project_id_async())

    def list_root_runs(self, limit: int = 20) -> list[RunSummary]:
        client = self._get_client()
        project = self._resolve_project()
        logger.info("Listing root runs (project=%s, limit=%s)", project, limit)
        start = time.monotonic()

        async def _fetch() -> list[Any]:
            project_id = await self._resolve_project_id_async()
            min_start = datetime.now(UTC) - timedelta(days=_LIST_MIN_START_DAYS)
            results: list[Any] = []
            async for trace in client.traces.query(
                project_id=project_id,
                selects=_LIST_SELECTS,
                min_start_time=min_start,
                page_size=min(limit, 1000),
            ):
                results.append(trace)
                if len(results) >= limit:
                    break
            return results

        traces = _run_async(_fetch())
        logger.info("Found %d root runs in %.2fs", len(traces), time.monotonic() - start)
        return [_trace_to_summary(t) for t in traces]

    def list_runs_by_type(self, run_type: str, limit: int = 20) -> list[RunSummary]:
        client = self._get_client()
        project = self._resolve_project()
        logger.info(
            "Listing runs by type (project=%s, run_type=%s, limit=%s)", project, run_type, limit
        )
        start = time.monotonic()

        async def _fetch() -> list[Any]:
            project_id = await self._resolve_project_id_async()
            min_start = datetime.now(UTC) - timedelta(days=_LIST_MIN_START_DAYS)
            results: list[Any] = []
            async for run in client.runs.query(
                project_ids=[project_id],
                run_type=run_type.upper(),
                selects=_LIST_SELECTS,
                min_start_time=min_start,
                page_size=min(limit, 1000),
            ):
                results.append(run)
                if len(results) >= limit:
                    break
            return results

        runs = _run_async(_fetch())
        logger.info("Found %d %s runs in %.2fs", len(runs), run_type, time.monotonic() - start)
        return [_sdk_run_to_summary(r) for r in runs]

    def fetch_trace(self, trace_id: str, project_id: str | None = None) -> Trace:
        """Download all runs of a trace and return a canonical Trace.

        Returns a Trace with sanitized=False. The caller is responsible for
        anonymizing before persistence.

        When ``project_id`` is provided, it is used to scope the query instead
        of the configured project name. This matters when the trace lives in a
        different project than the default one (e.g. when resolving from a run
        ID that belongs to another project).
        """
        client = self._get_client()
        if project_id:
            logger.info("Fetching trace %s (project_id=%s)", trace_id, project_id)
        else:
            project = self._resolve_project()
            logger.info("Fetching trace %s (project=%s)", trace_id, project)

        async def _fetch() -> list[Any]:
            pid = project_id or await self._resolve_project_id_async()
            response = await client.traces.list_runs(
                trace_id=trace_id,
                project_id=pid,
                selects=_FULL_SELECTS,
            )
            return response.items or []

        start = time.monotonic()
        sdk_runs = _run_async(_fetch())
        sdk_runs.sort(key=lambda r: r.dotted_order or "")
        logger.info(
            "Fetched %d runs for trace %s in %.2fs",
            len(sdk_runs),
            trace_id,
            time.monotonic() - start,
        )

        canonical_runs = [_sdk_run_to_canonical(r) for r in sdk_runs]
        return Trace(
            trace_id=trace_id,
            runs=canonical_runs,
            sanitized=False,
            source="langsmith",
        )

    def resolve_trace_id(self, run_id: str) -> RunResolution:
        """Resolve a run ID to its parent trace ID and project.

        Uses the LangSmith SDK's read_run to fetch the run metadata and
        extract its trace_id and session_id (the project UUID). This is
        useful when the user only has a run ID (e.g. from a LangSmith trace
        URL) and needs the trace ID to fetch the full trace.

        Returns a RunResolution with both the trace_id and the project_id
        (session_id) so the caller can fetch the trace from the correct
        project, even when it differs from the configured default.

        TODO: Migrate to runs.retrieve (SmithDB). The new API requires
        project_id as input, but this method's purpose is to discover the
        project_id from a bare run_id — a chicken-and-egg problem. Keep
        read_run (legacy, deprecated end of July 2026, removed 31 Jan 2027)
        until a SmithDB API resolves run_id without a known project_id.
        """
        client = self._get_client()
        logger.info("Resolving run %s to trace_id", run_id)
        start = time.monotonic()
        sdk_run = client.read_run(run_id)
        trace_id = str(sdk_run.trace_id)
        project_id = getattr(sdk_run, "session_id", None)
        project_id = str(project_id) if project_id else None
        logger.info(
            "Resolved run %s -> trace_id %s (project_id=%s) in %.2fs",
            run_id,
            trace_id,
            project_id,
            time.monotonic() - start,
        )
        return RunResolution(trace_id=trace_id, project_id=project_id)

    def list_projects(self, limit: int = 50) -> list[ProjectSummary]:
        """List accessible LangSmith projects, most recent first."""
        client = self._get_client()
        logger.info("Listing projects (limit=%s)", limit)
        start = time.monotonic()
        projects = list(client.list_projects(limit=limit))
        logger.info("Found %d projects in %.2fs", len(projects), time.monotonic() - start)
        return [
            ProjectSummary(
                id=str(p.id),
                name=p.name,
                run_count=getattr(p, "run_count", None),
            )
            for p in projects
        ]

    def pull_prompt(
        self, name: str, tag: str | None = None, workspace_id: str | None = None
    ) -> str:
        """Pull a prompt from LangSmith Prompt Hub and return its template text.

        Args:
            name: The prompt name without tag (e.g. "react_agent").
            tag: The environment tag (e.g. "prod", "staging", "test").
                 If None, pulls the latest.
            workspace_id: Optional LangSmith workspace UUID for non-default
                 workspaces (e.g. Workspace 2). If None, uses the default workspace.

        Returns:
            The prompt template string. For ChatPromptTemplate prompts (the
            common case), all message templates are joined with newlines.

        Raises:
            ImportError: If langsmith is not installed.
            RuntimeError: If the prompt is not found or the manifest cannot
                be parsed.
        """
        # Use a workspace-specific client when workspace_id is provided,
        # since workspace_id is a Client constructor parameter, not a
        # per-call parameter in the LangSmith Python SDK.
        if workspace_id:
            client = _get_client(workspace_id=workspace_id)
        else:
            client = self._get_client()
        full_name = f"{name}:{tag}" if tag else name
        logger.info("Pulling prompt %s (workspace=%s)", full_name, workspace_id or "default")
        commit = client.pull_prompt_commit(full_name, include_model=False)
        return _extract_template_from_manifest(commit.manifest)


def _extract_template_from_manifest(manifest: Any) -> str:
    """Extract the template text from a LangSmith prompt manifest dict.

    LangSmith stores prompts as serialized LangChain objects. The manifest is
    a dict with ``lc``, ``type``, ``id``, ``kwargs``. We extract the template
    string(s) without needing langchain_core installed.

    Supports:
    - ChatPromptTemplate: joins all message templates with newlines.
    - PromptTemplate: returns the single template string.
    - Fallback: JSON-dumps the manifest if the structure is unrecognized.
    """
    if not isinstance(manifest, dict):
        return json.dumps(manifest, indent=2, default=str)

    lc_id = manifest.get("id", [])
    kwargs = manifest.get("kwargs", {})

    # ChatPromptTemplate: ["langchain", "prompts", "chat", "ChatPromptTemplate"]
    # or similar — has a "messages" list.
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        templates: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_kwargs = msg.get("kwargs", {})
            # SystemMessagePromptTemplate / HumanMessagePromptTemplate etc.
            prompt = msg_kwargs.get("prompt")
            if isinstance(prompt, dict):
                t = prompt.get("kwargs", {}).get("template")
                if isinstance(t, str):
                    templates.append(t)
            # Some message formats store template directly
            elif isinstance(msg_kwargs.get("template"), str):
                templates.append(msg_kwargs["template"])
        if templates:
            return "\n".join(templates)

    # Plain PromptTemplate: ["langchain", "prompts", "prompt", "PromptTemplate"]
    template = kwargs.get("template")
    if isinstance(template, str):
        return template

    # Fallback: serialize the manifest so the user can inspect it.
    logger.warning("Unrecognized prompt manifest structure (id=%s)", lc_id)
    return json.dumps(manifest, indent=2, default=str)
