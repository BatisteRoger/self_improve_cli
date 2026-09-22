"""Discovery and ingestion commands — list, list-runs, list-projects, fetch.

These are the only commands that talk to LangSmith (besides `prompt pull`).
Everything else operates on locally stored sanitized traces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from self_improve_cli.cli.common import (
    EXIT_OK,
    _get_store,
    _output,
    add_common_opts,
)
from self_improve_cli.privacy import (
    AnonymizerBackend,
    anonymize_trace,
    is_presidio_available,
)
from self_improve_cli.representations import write_ter


def _cmd_list(args: argparse.Namespace) -> int:
    from self_improve_cli.sources.langsmith import LangSmithSource

    source = LangSmithSource(project_name=args.project)
    summaries = source.list_root_runs(limit=args.limit)
    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "id": s.id,
                        "trace_id": s.trace_id,
                        "name": s.name,
                        "run_type": s.run_type.value,
                        "status": s.status,
                        "start_time": s.start_time,
                        "total_tokens": s.total_tokens,
                        "error": bool(s.error),
                    }
                    for s in summaries
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for s in summaries:
            print(
                f"{s.start_time}  {s.name}  status={s.status} "
                f"tokens={s.total_tokens} error={bool(s.error)}  trace_id={s.trace_id}"
            )
    return EXIT_OK


def _cmd_list_runs(args: argparse.Namespace) -> int:
    from self_improve_cli.sources.langsmith import LangSmithSource

    source = LangSmithSource(project_name=args.project)
    summaries = source.list_runs_by_type(run_type=args.type, limit=args.limit)
    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "id": s.id,
                        "trace_id": s.trace_id,
                        "name": s.name,
                        "run_type": s.run_type.value,
                        "status": s.status,
                        "start_time": s.start_time,
                        "total_tokens": s.total_tokens,
                    }
                    for s in summaries
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for s in summaries:
            print(
                f"{s.start_time}  [{s.run_type.value}] {s.name}  status={s.status} "
                f"tokens={s.total_tokens}  trace_id={s.trace_id}  run_id={s.id}"
            )
    return EXIT_OK


def _cmd_list_projects(args: argparse.Namespace) -> int:
    """List accessible LangSmith projects."""
    from self_improve_cli.sources.langsmith import LangSmithSource

    source = LangSmithSource()
    projects = source.list_projects(limit=args.limit)
    if args.format == "json":
        print(
            json.dumps(
                [{"id": p.id, "name": p.name, "run_count": p.run_count} for p in projects],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for p in projects:
            runs = f"  runs={p.run_count}" if p.run_count is not None else ""
            print(f"{p.name}  id={p.id}{runs}")
    return EXIT_OK


def _resolve_anonymizer(args: argparse.Namespace) -> AnonymizerBackend:
    """Resolve the anonymizer backend: CLI flag overrides env var, env var overrides 'auto'.

    Valid values: "auto", "presidio", "regex".
    """
    cli_val = getattr(args, "anonymizer", None)
    if cli_val:
        return cli_val  # type: ignore[return-value]
    env_val = os.environ.get("SELFIIMPROVE_ANONYMIZER", "").strip().lower()
    if env_val in ("auto", "presidio", "regex"):
        return env_val  # type: ignore[return-value]
    return "auto"


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch a trace, anonymize it, and persist the sanitized version."""
    from self_improve_cli.sources.langsmith import LangSmithSource

    source = LangSmithSource(project_name=args.project)

    project_id: str | None = None

    # Resolve run_id -> trace_id if the user passed --from-run.
    if getattr(args, "from_run", False):
        resolution = source.resolve_trace_id(args.trace_id)
        trace_id = resolution.trace_id
        project_id = resolution.project_id
        print(
            f"Resolved run {args.trace_id} -> trace {trace_id}"
            + (f" (project_id={project_id})" if project_id else ""),
            file=sys.stderr,
        )
    else:
        trace_id = args.trace_id

    trace = source.fetch_trace(trace_id, project_id=project_id)

    # Warn on empty traces — likely a project mismatch.
    if not trace.runs:
        hint = (
            "No runs found. The trace may live in a different project than the "
            "configured one. Try `self-improve list-projects` to discover project "
            "names, then `self-improve fetch <trace_id> --project <name>`."
        )
        if project_id:
            hint = (
                f"No runs found for trace {trace_id} in project_id={project_id}. "
                "The trace may be empty or access may be restricted."
            )
        print(f"Warning: {hint}", file=sys.stderr)

    # Save raw BEFORE anonymization (anonymize_trace mutates in place).
    store = _get_store(args)
    if args.keep_raw:
        store.save_raw(trace)

    # Anonymize before persistence (privacy boundary).
    backend = _resolve_anonymizer(args)
    trace, report = anonymize_trace(trace, backend=backend)

    sanitized_path = store.save_sanitized(trace)

    stats = write_ter(trace_id, trace.runs, store)

    result: dict[str, Any] = {
        "trace_id": trace_id,
        "runs": len(trace.runs),
        "sanitized": True,
        "sanitization_report": report.to_dict(),
        "anonymizer_backend": backend,
        "presidio_available": is_presidio_available(),
        "sanitized_path": str(sanitized_path),
        "raw_saved": args.keep_raw,
        "ter_stats": stats,
    }
    if getattr(args, "from_run", False):
        result["resolved_from_run"] = args.trace_id
    _output(result, args)
    return EXIT_OK


def register(sub: argparse._SubParsersAction) -> None:
    """Register discovery commands on the given subparsers action."""
    # list
    p = sub.add_parser("list", help="L0: recent root runs (one per trace)")
    p.add_argument("--project", default=None, help="LangSmith project name")
    p.add_argument("--limit", type=int, default=10)
    add_common_opts(p)
    p.set_defaults(func=_cmd_list)

    # list-runs
    p = sub.add_parser("list-runs", help="L0: recent runs of a type, project-wide")
    p.add_argument("--project", default=None, help="LangSmith project name")
    p.add_argument("--type", choices=["llm", "tool", "chain", "other"], required=True)
    p.add_argument("--limit", type=int, default=10)
    add_common_opts(p)
    p.set_defaults(func=_cmd_list_runs)

    # list-projects
    p = sub.add_parser("list-projects", help="L0: list accessible LangSmith projects")
    p.add_argument("--limit", type=int, default=50)
    add_common_opts(p)
    p.set_defaults(func=_cmd_list_projects)

    # fetch
    p = sub.add_parser("fetch", help="Download a trace, anonymize it, and build TER")
    p.add_argument("--project", default=None, help="LangSmith project name")
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="Also save a raw (unanonymized) copy locally. Use with caution.",
    )
    p.add_argument(
        "--from-run",
        action="store_true",
        help="Treat the positional argument as a run ID and resolve it to its "
        "parent trace ID before fetching. Tries the configured project first "
        "(SmithDB); falls back to a legacy global lookup if the run is not in "
        "the configured project. Use --project to target a different project.",
    )
    p.add_argument(
        "--anonymizer",
        choices=["auto", "presidio", "regex"],
        default=None,
        help="Anonymization backend: 'auto' (presidio if installed, else regex), "
        "'presidio' (force, errors if unavailable), 'regex' (skip presidio, faster). "
        "Overrides SELFIIMPROVE_ANONYMIZER env var. Default: auto.",
    )
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_fetch)
