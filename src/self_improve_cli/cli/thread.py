"""`thread` command: list the turns (traces) of one LangSmith conversation.

A conversation (thread) spans several traces, one per turn. This view maps a
thread ID or a LangSmith thread URL to its turns, oldest first, with an
anonymized excerpt of each turn's input and output. Nothing is persisted:
the excerpts are anonymized in memory, with the same pipeline and settings
as `fetch`, before they are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

from self_improve_cli.cli.common import EXIT_ERROR, EXIT_OK
from self_improve_cli.cli.discovery import _resolve_anonymizer, _resolve_structural_policy
from self_improve_cli.domain import Run, Trace
from self_improve_cli.privacy import anonymize_trace

_EXCERPT_CHARS = 120


def _excerpt(value: Any) -> str:
    """One-line, bounded excerpt of an anonymized preview."""
    text = " ".join(str(value or "").split())
    if len(text) > _EXCERPT_CHARS:
        return text[:_EXCERPT_CHARS] + "..."
    return text


def _latency_s(run: Run) -> float | None:
    if not run.start_time or not run.end_time:
        return None
    start = datetime.fromisoformat(run.start_time)
    end = datetime.fromisoformat(run.end_time)
    return round((end - start).total_seconds(), 2)


def _turn_record(index: int, run: Run) -> dict[str, Any]:
    return {
        "turn": index,
        "trace_id": run.trace_id,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "status": run.status,
        "latency_s": _latency_s(run),
        "total_tokens": run.total_tokens,
        "error": bool(run.error),
        "error_excerpt": _excerpt(run.error) if run.error else None,
        "input_excerpt": _excerpt(run.inputs.get("preview")),
        "output_excerpt": _excerpt(run.outputs.get("preview")),
    }


def _cell(text: str | None) -> str:
    return (text or "").replace("|", "/")


def _markdown(result: dict[str, Any]) -> str:
    turns = result["turns"]
    lines = [
        f"# Thread {result['thread_id']}",
        f"project: {result['project_name']} (id={result['project_id']})  turns: {len(turns)}"
        + (f"  [truncated at {len(turns)}]" if result["truncated"] else ""),
        "Excerpts: LangSmith previews of the root input/output, anonymized "
        f"({result['anonymizer_backend']}), cut to {_EXCERPT_CHARS} chars.",
        "",
        "| # | start (UTC) | trace_id | status | latency_s | tokens | error | input | output |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in turns:
        start = (t["start_time"] or "?")[:19]
        latency = "?" if t["latency_s"] is None else f"{t['latency_s']:.1f}"
        tokens = "n/a" if t["total_tokens"] is None else str(t["total_tokens"])
        lines.append(
            f"| {t['turn']} | {start} | {t['trace_id']} | {t['status']} | {latency} "
            f"| {tokens} | {'yes' if t['error'] else 'no'} "
            f"| {_cell(t['input_excerpt'])} | {_cell(t['output_excerpt'])} |"
        )
    lines += ["", "Next:"] + [f"  {cmd}" for cmd in result["next"]]
    return "\n".join(lines)


def _cmd_thread(args: argparse.Namespace) -> int:
    from self_improve_cli.sources.langsmith import LangSmithSource
    from self_improve_cli.sources.langsmith_url import is_url, parse_langsmith_thread_url

    # Resolve privacy settings before any network call: a bad spec fails closed.
    backend = _resolve_anonymizer(args)
    drop_keys, value_maps = _resolve_structural_policy()

    thread_id, project_id = args.thread, None
    if is_url(args.thread):
        ref = parse_langsmith_thread_url(args.thread)
        thread_id, project_id = ref.thread_id, ref.project_id

    source = LangSmithSource(project_name=args.project)
    listing = source.list_thread_turns(thread_id, project_id=project_id, limit=args.limit)
    if not listing.runs:
        print(
            f"Error: no traces found for thread {thread_id} in project "
            f"'{listing.project_name}' (id={listing.project_id}). LangSmith groups traces "
            "into a thread by the session_id, thread_id or conversation_id metadata key. "
            "Check the project: paste the thread URL from the LangSmith UI (it carries "
            "the project), or pass --project <name>.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    # One anonymization pass over all turns: placeholders stay stable across turns.
    trace = Trace(trace_id=f"thread:{thread_id}", runs=listing.runs, source="langsmith")
    trace, report = anonymize_trace(
        trace, backend=backend, drop_keys=drop_keys, value_maps=value_maps
    )
    turns = [_turn_record(i, run) for i, run in enumerate(trace.runs, start=1)]
    project_opt = f" --project '{listing.project_name}'" if listing.project_name else ""
    result: dict[str, Any] = {
        "thread_id": thread_id,
        "project_id": listing.project_id,
        "project_name": listing.project_name,
        "truncated": listing.truncated,
        "anonymizer_backend": backend,
        "sanitization_report": report.to_dict(),
        "turns": turns,
        "next": [f"self-improve fetch <trace_id>{project_opt}"],
    }
    if args.format == "json":
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    else:
        print(_markdown(result))
    return EXIT_OK


def register(sub: argparse._SubParsersAction) -> None:
    """Register the `thread` command on the given subparsers action."""
    p = sub.add_parser(
        "thread",
        help="L0: list the turns (traces) of a LangSmith conversation, with excerpts",
    )
    p.add_argument(
        "thread",
        help="Thread ID, or a LangSmith URL carrying peekedConversationId=<thread_id> "
        "(the URL also gives the project).",
    )
    p.add_argument("--project", default=None, help="LangSmith project name")
    p.add_argument("--limit", type=int, default=200, help="Maximum number of turns (default: 200)")
    p.add_argument(
        "--anonymizer",
        choices=["auto", "presidio", "regex"],
        default=None,
        help="Anonymization backend for the excerpts, as for fetch. "
        "Overrides SELF_IMPROVE_ANONYMIZER env var. Default: auto.",
    )
    p.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p.set_defaults(func=_cmd_thread)
