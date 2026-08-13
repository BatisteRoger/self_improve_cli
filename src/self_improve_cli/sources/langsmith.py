"""LangSmith trace source adapter.

Normalizes LangSmith SDK Run objects into the canonical domain model.
SDK-specific objects stop here — nothing downstream imports langsmith.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import time
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

_LIST_FIELDS = [
    "id",
    "trace_id",
    "name",
    "run_type",
    "status",
    "start_time",
    "end_time",
    "error",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
]


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
            "LANGSMITH_API_KEY is not set. Copy .env.example to .env and fill it in."
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
        return RunType(raw)
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

    return Run(
        id=str(run_dict.get("id", "")),
        trace_id=str(run_dict.get("trace_id", "")),
        run_type=_parse_run_type(run_dict.get("run_type")),
        name=run_dict.get("name", ""),
        parent_run_id=run_dict.get("parent_run_id"),
        dotted_order=run_dict.get("dotted_order"),
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

    def list_root_runs(self, limit: int = 20) -> list[RunSummary]:
        client = self._get_client()
        project = self._resolve_project()
        logger.info("Listing root runs (project=%s, limit=%s)", project, limit)
        start = time.monotonic()
        runs = list(
            client.list_runs(
                project_name=project,
                is_root=True,
                limit=limit,
                select=_LIST_FIELDS,
            )
        )
        logger.info("Found %d root runs in %.2fs", len(runs), time.monotonic() - start)
        return [_sdk_run_to_summary(r) for r in runs]

    def list_runs_by_type(self, run_type: str, limit: int = 20) -> list[RunSummary]:
        client = self._get_client()
        project = self._resolve_project()
        logger.info(
            "Listing runs by type (project=%s, run_type=%s, limit=%s)", project, run_type, limit
        )
        start = time.monotonic()
        runs = list(
            client.list_runs(
                project_name=project,
                run_type=run_type,
                limit=limit,
                select=_LIST_FIELDS,
            )
        )
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
            query_kwargs: dict[str, Any] = {"project_id": project_id, "trace_id": trace_id}
        else:
            project = self._resolve_project()
            logger.info("Fetching trace %s (project=%s)", trace_id, project)
            query_kwargs = {"project_name": project, "trace_id": trace_id}
        start = time.monotonic()
        sdk_runs = list(client.list_runs(**query_kwargs))
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
