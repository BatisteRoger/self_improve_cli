"""Main CLI entry point.

Agent-first design:
- Default output is compact Markdown, bounded for model context windows.
- --format json gives a versioned schema for programmatic consumers.
- stdout for results, stderr for diagnostics.
- Stable exit codes: 0 success, 1 recoverable error, 2 usage error.
- Network operations are explicit (fetch). Analysis commands are offline.
- Privacy is enforced at fetch time: traces are anonymized before persistence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from self_improve_cli.cli.logging_setup import setup_logging
from self_improve_cli.privacy import anonymize_trace, is_presidio_available
from self_improve_cli.representations import build_narrative, build_skeleton, run_detail, write_ter
from self_improve_cli.storage import TraceStore

# Force UTF-8 on stdout/stderr for cross-platform Unicode support.
# Without this, Windows cp1252 crashes on characters like ->, e-acute in trace content.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

EPILOG = (
    "Privacy: traces are anonymized on fetch by default. "
    "Use --keep-raw to also retain a local raw copy (never shared)."
)

# Exit codes
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("self-improve-cli")
except Exception:  # noqa: BLE001
    __version__ = "0.0.0"


def _load_env() -> None:
    """Load .env from cwd if it exists."""
    load_dotenv(Path(".env"))


def _get_store(args: argparse.Namespace) -> TraceStore:
    data_root = Path(args.data_dir) if args.data_dir else Path("data")
    return TraceStore(data_root=data_root, keep_raw=args.keep_raw)


def _output(data: Any, args: argparse.Namespace) -> None:
    """Print results in the requested format (markdown or json)."""
    if args.format == "json":
        if isinstance(data, str):
            print(json.dumps({"content": data}, ensure_ascii=False))
        else:
            print(json.dumps(data, indent=2, default=str, ensure_ascii=False))
    else:
        if isinstance(data, str):
            print(data)
        else:
            print(json.dumps(data, indent=2, default=str, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


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


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch a trace, anonymize it, and persist the sanitized version."""
    from self_improve_cli.sources.langsmith import LangSmithSource

    source = LangSmithSource(project_name=args.project)
    trace = source.fetch_trace(args.trace_id)

    # Save raw BEFORE anonymization (anonymize_trace mutates in place).
    store = _get_store(args)
    if args.keep_raw:
        store.save_raw(trace)

    # Anonymize before persistence (privacy boundary).
    trace, report = anonymize_trace(trace)

    sanitized_path = store.save_sanitized(trace)

    stats = write_ter(args.trace_id, trace.runs, store)

    result = {
        "trace_id": args.trace_id,
        "runs": len(trace.runs),
        "sanitized": True,
        "sanitization_report": report.to_dict(),
        "presidio_available": is_presidio_available(),
        "sanitized_path": str(sanitized_path),
        "raw_saved": args.keep_raw,
        "ter_stats": stats,
    }
    _output(result, args)
    return EXIT_OK


def _cmd_skeleton(args: argparse.Namespace) -> int:
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    content = build_skeleton(trace.runs)
    _output(content, args)
    return EXIT_OK


def _cmd_narrative(args: argparse.Namespace) -> int:
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    content = build_narrative(trace.runs)
    _output(content, args)
    return EXIT_OK


def _cmd_tool_metrics(args: argparse.Namespace) -> int:
    from self_improve_cli.metrics.tool_metrics import build_tool_metrics

    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    content = build_tool_metrics(trace.runs)
    _output(content, args)
    return EXIT_OK


def _cmd_context_metrics(args: argparse.Namespace) -> int:
    from self_improve_cli.metrics.context_metrics import build_context_metrics

    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    content = build_context_metrics(trace.runs)
    _output(content, args)
    return EXIT_OK


def _cmd_run_detail(args: argparse.Namespace) -> int:
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    content = run_detail(trace.runs, args.run_id)
    _output(content, args)
    return EXIT_OK


def _cmd_info(args: argparse.Namespace) -> int:
    """Show info about a saved trace (sanitization status, stats)."""
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    result = {
        "trace_id": trace.trace_id,
        "sanitized": trace.sanitized,
        "sanitization_report": trace.sanitization_report,
        "source": trace.source,
        "schema_version": trace.schema_version,
        "run_count": len(trace.runs),
    }
    _output(result, args)
    return EXIT_OK


def _cmd_skill(args: argparse.Namespace) -> int:
    """Print the navigation workflow from SKILL.md."""
    from importlib.resources import files

    skill_path = Path(str(files("self_improve_cli"))) / "SKILL.md"
    if not skill_path.exists():
        print("SKILL.md not found.", file=sys.stderr)
        return EXIT_ERROR
    print(skill_path.read_text(encoding="utf-8"))
    return EXIT_OK


def _cmd_init(args: argparse.Namespace) -> int:
    """Create a .env file from .env.example if it doesn't exist."""
    env_path = Path(".env")
    if env_path.exists():
        print(".env already exists. Edit it to fill in your values.", file=sys.stderr)
        return EXIT_ERROR
    from importlib.resources import files

    example_path = Path(str(files("self_improve_cli"))) / ".env.example"
    if not example_path.exists():
        print(".env.example not found.", file=sys.stderr)
        return EXIT_ERROR
    env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
    print("Created .env from .env.example. Edit it to fill in your LangSmith API key.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self-improve",
        description="Privacy-first CLI for inspecting AI-agent traces.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # Global options that apply to most commands
    def add_common_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--format",
            choices=["markdown", "json"],
            default="markdown",
            help="Output format (default: markdown)",
        )
        p.add_argument("--data-dir", default=None, help="Local data root (default: ./data)")

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

    # fetch
    p = sub.add_parser("fetch", help="Download a trace, anonymize it, and build TER")
    p.add_argument("--project", default=None, help="LangSmith project name")
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="Also save a raw (unanonymized) copy locally. Use with caution.",
    )
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_fetch)

    # skeleton
    p = sub.add_parser("skeleton", help="L1: one line per significant run")
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_skeleton)

    # narrative
    p = sub.add_parser("narrative", help="L2: trace story with message deltas")
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_narrative)

    # tool-metrics
    p = sub.add_parser("tool-metrics", help="L1: tool call patterns & efficiency signals")
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_tool_metrics)

    # context-metrics
    p = sub.add_parser("context-metrics", help="L1: token decomposition & growth curve")
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_context_metrics)

    # run-detail
    p = sub.add_parser("run-detail", help="L3: full prompts/outputs of one run")
    p.add_argument("trace_id")
    p.add_argument("run_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_run_detail)

    # info
    p = sub.add_parser("info", help="Show info about a saved trace")
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_info)

    # skill
    p = sub.add_parser("skill", help="Print the recommended navigation workflow")
    p.set_defaults(func=_cmd_skill)

    # init
    p = sub.add_parser("init", help="Create a .env file from .env.example")
    p.set_defaults(func=_cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env()
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    # keep_raw is only valid for fetch
    if not hasattr(args, "keep_raw"):
        args.keep_raw = False

    # Enforce --project exclusivity: if .env sets LANGSMITH_PROJECT, refuse --project.
    env_project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT")
    if env_project and getattr(args, "project", None) is not None:
        print(
            "Error: --project is not allowed when .env sets LANGSMITH_PROJECT. Edit .env instead.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Hint: run `self-improve fetch <trace_id>` first.", file=sys.stderr)
        return EXIT_ERROR
    except (RuntimeError, ImportError, ValueError, PermissionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
