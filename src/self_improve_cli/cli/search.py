"""`search` command: how often does a text or pattern appear in recent run outputs?

L0 discovery, read-only: it scans a project's runs over a time window, matches
their output text, and reports matched / scanned (the frequency) with one line
per match. Nothing is written to disk. Raw run content stays in memory; only
anonymized excerpts are printed (same anonymizer and settings as `fetch`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from self_improve_cli.cli.common import EXIT_OK, EXIT_USAGE, _output, add_common_opts
from self_improve_cli.cli.discovery import _resolve_anonymizer, _resolve_structural_policy
from self_improve_cli.domain import Run, RunType, Trace
from self_improve_cli.privacy import anonymize_trace

# Raw context kept around a match before anonymization, then the printed
# context after it. The wider raw window keeps a value cut at its edge out of
# the printed excerpt: detection runs on whole values well inside the window.
_RAW_CONTEXT = 200
_EXCERPT_CONTEXT = 80

Matcher = Callable[[str], "tuple[int, int] | None"]


def output_text(run: Run) -> str:
    """Searchable output of a run: message text plus tool-call args, else raw outputs."""
    msg = run.output_message
    if msg is not None:
        parts = [msg.text] + [
            json.dumps({tc.name: tc.args}, ensure_ascii=False) for tc in msg.tool_calls
        ]
        return "\n".join(p for p in parts if p)
    return json.dumps(run.outputs, ensure_ascii=False) if run.outputs else ""


def build_matcher(text: str | None, pattern: re.Pattern[str] | None) -> Matcher:
    """Span of the first match; both criteria must match when both are given.

    `text` is a case-insensitive substring (the server-side full-text search
    is not case-sensitive either). The excerpt centers on the regex match
    when there is one.
    """
    needle = text.casefold() if text else None

    def match(haystack: str) -> tuple[int, int] | None:
        span: tuple[int, int] | None = None
        if needle is not None:
            # Offsets are approximate if casefold changed a length; only the excerpt uses them.
            idx = haystack.casefold().find(needle)
            if idx < 0:
                return None
            span = (idx, idx + len(needle))
        if pattern is not None:
            m = pattern.search(haystack)
            if m is None:
                return None
            span = (m.start(), m.end())
        return span

    return match


def _model_name(run: Run) -> str | None:
    meta = run.extra.get("metadata") or (run.extra.get("extra") or {}).get("metadata") or {}
    name = meta.get("ls_model_name") if isinstance(meta, dict) else None
    return str(name) if name else None


def anonymized_excerpt(
    run: Run,
    haystack: str,
    span: tuple[int, int],
    match: Matcher,
    anonymize: Callable[[Run], Run],
) -> str:
    """One-line excerpt around the match, anonymized like `fetch` before it is cut."""
    start, end = span
    lo, hi = max(0, start - _RAW_CONTEXT), min(len(haystack), end + _RAW_CONTEXT)
    probe = Run(
        id=run.id,
        trace_id=run.trace_id,
        run_type=run.run_type,
        name=run.name,
        outputs={"excerpt": haystack[lo:hi]},
    )
    clean = str(anonymize(probe).outputs.get("excerpt", ""))
    # The match may itself have been redacted: then center on the window.
    found = match(clean)
    if found is None:
        mid = len(clean) // 2
        found = (mid, mid)
    a = max(0, found[0] - _EXCERPT_CONTEXT)
    b = min(len(clean), found[1] + _EXCERPT_CONTEXT)
    body = " ".join(clean[a:b].split())
    return (
        ("..." if a > 0 or lo > 0 else "")
        + body
        + ("..." if b < len(clean) or hi < len(haystack) else "")
    )


def _parse_metadata(items: list[str]) -> list[tuple[str, str]] | str:
    pairs: list[tuple[str, str]] = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            return f"--metadata expects key=value, got {item!r}."
        pairs.append((key.strip(), value))
    return pairs


def _cmd_search(args: argparse.Namespace) -> int:
    from self_improve_cli.sources.langsmith import LangSmithSource, build_search_filter

    usage = None
    pattern: re.Pattern[str] | None = None
    if not args.text and not args.regex:
        usage = "pass --text and/or --regex."
    elif args.since <= 0 or args.limit < 0 or args.max_scan <= 0:
        usage = "--since and --max-scan must be positive, --limit zero or more."
    elif args.regex:
        try:
            pattern = re.compile(args.regex)
        except re.error as exc:
            usage = f"invalid --regex: {exc}."
    metadata = _parse_metadata(args.metadata or [])
    if isinstance(metadata, str):
        usage = metadata
    if usage or isinstance(metadata, str):  # second test narrows the type
        print(f"Error: {usage}", file=sys.stderr)
        return EXIT_USAGE

    # Resolve privacy settings before any network call: a bad spec fails closed.
    backend = _resolve_anonymizer(args)
    drop_keys, value_maps = _resolve_structural_policy()

    def anonymize(run: Run) -> Run:
        trace = Trace(trace_id=run.trace_id, runs=[run], sanitized=False, source="langsmith")
        anonymize_trace(trace, backend=backend, drop_keys=drop_keys, value_maps=value_maps)
        return trace.runs[0]

    match = build_matcher(args.text, pattern)
    filter_expr = build_search_filter(args.text, metadata)
    source = LangSmithSource(project_name=args.project)

    scanned = 0
    hits: list[tuple[Run, str, tuple[int, int]]] = []
    matched = 0

    def on_run(run: Run) -> bool:
        nonlocal scanned, matched
        scanned += 1
        haystack = output_text(run)
        span = match(haystack)
        if span is not None:
            matched += 1
            if len(hits) < args.limit:
                hits.append((run, haystack, span))
        return scanned < args.max_scan

    started = time.monotonic()
    exhausted = source.scan_runs(
        run_type=args.run_type,
        since_days=args.since,
        filter_expr=filter_expr,
        on_run=on_run,
    )
    elapsed = round(time.monotonic() - started, 1)

    matches = [
        {
            "start_time": run.start_time,
            "trace_id": run.trace_id,
            "run_id": run.id,
            "model": _model_name(run),
            "excerpt": anonymized_excerpt(run, haystack, span, match, anonymize),
        }
        for run, haystack, span in hits
    ]
    result: dict[str, Any] = {
        "project": source.project_name,
        "run_type": args.run_type,
        "since_days": args.since,
        "server_filter": filter_expr,
        "scanned": scanned,
        "matched": matched,
        "complete": exhausted,
        "shown": len(matches),
        "elapsed_s": elapsed,
        "anonymizer_backend": backend,
        "matches": matches,
    }
    if not exhausted:
        print(
            f"Warning: stopped after --max-scan {args.max_scan} runs; counts cover "
            "the most recent runs only. Raise --max-scan or narrow --since/--metadata.",
            file=sys.stderr,
        )
    if args.format == "json":
        _output(result, args)
    else:
        print(_markdown(result))
    return EXIT_OK


def _markdown(r: dict[str, Any]) -> str:
    pct = f" ({100 * r['matched'] / r['scanned']:.2f}%)" if r["scanned"] else ""
    lines = [
        f"# search: {r['project']} [{r['run_type']}] last {r['since_days']} days",
        f"matched {r['matched']} of {r['scanned']} runs scanned{pct}"
        + ("" if r["complete"] else " (INCOMPLETE: --max-scan reached)")
        + f", {r['elapsed_s']}s",
    ]
    if r["server_filter"]:
        lines.append(
            f"scanned = runs returned by the server filter: {r['server_filter']}"
            " (full-text search also covers inputs; matches are confirmed in outputs)"
        )
    if r["matched"] > r["shown"]:
        lines.append(f"showing {r['shown']} of {r['matched']} matches (raise --limit for more)")
    lines.append("")
    for m in r["matches"]:
        model = f"  model={m['model']}" if m["model"] else ""
        lines.append(
            f"{m['start_time']}  trace_id={m['trace_id']}  run_id={m['run_id']}{model}\n"
            f"    {m['excerpt']}"
        )
    if r["matches"]:
        lines += ["", f"next: self-improve fetch {r['matches'][0]['trace_id']}"]
    return "\n".join(lines)


def register(sub: argparse._SubParsersAction) -> None:
    """Register the search command on the given subparsers action."""
    p = sub.add_parser(
        "search",
        help="L0: runs whose output matches a text or regex, with frequency (matched/scanned)",
    )
    p.add_argument("--project", default=None, help="LangSmith project name")
    p.add_argument("--since", type=int, default=14, help="Window in days (default: 14)")
    p.add_argument(
        "--run-type",
        choices=[t.value for t in RunType if t is not RunType.OTHER],
        default="llm",
        help="Run type to scan (default: llm)",
    )
    p.add_argument(
        "--text",
        default=None,
        help="Case-insensitive text: narrows server-side (full-text search), then "
        "confirmed in the output text",
    )
    p.add_argument(
        "--regex", default=None, help="Python regex matched client-side on the output text"
    )
    p.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        help="Server-side metadata equality filter (repeatable, all must match)",
    )
    p.add_argument(
        "--limit", type=int, default=20, help="Matches shown (default: 20); all are counted"
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=20000,
        help="Stop after scanning this many runs, newest first (default: 20000)",
    )
    p.add_argument(
        "--anonymizer",
        choices=["auto", "presidio", "regex"],
        default=None,
        help="Excerpt anonymizer, as for fetch (default: SELF_IMPROVE_ANONYMIZER or auto)",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_search)
