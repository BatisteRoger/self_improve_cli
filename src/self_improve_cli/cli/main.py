"""Main CLI entry point.

Agent-first design:
- Default output is compact Markdown, bounded for model context windows.
- --format json gives a versioned schema for programmatic consumers.
- stdout for results, stderr for diagnostics.
- Stable exit codes: 0 success, 1 recoverable error, 2 usage error.
- Network operations are explicit (fetch). Analysis commands are offline.
- Privacy is enforced at fetch time: traces are anonymized before persistence.

This module is deliberately thin: command handlers and their argparse
registration live in per-family modules — `discovery` (list/fetch), `views`
(trace representations), `metrics`, `records` (info/assess/finding),
`system` (skill/init/doctor), `prompts`, `ati`. Shared plumbing (exit codes,
store construction, output formatting) lives in `common`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from self_improve_cli.cli import ati, discovery, metrics, prompts, records, system, views
from self_improve_cli.cli.common import EXIT_ERROR
from self_improve_cli.cli.logging_setup import setup_logging

# Force UTF-8 on stdout/stderr for cross-platform Unicode support.
# Without this, Windows cp1252 crashes on characters like →, —, é in
# authored CLI output (run-detail, context-at, target-timeline) and in
# trace content surfaced by analysis commands. See SLN-40.


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 if the default encoding cannot handle Unicode.

    Best-effort: streams without ``reconfigure`` (e.g. StringIO in tests,
    captured streams) are skipped silently. A reconfigure that raises is
    swallowed so the CLI never crashes at startup over an encoding fix.
    """
    for _stream in (sys.stdout, sys.stderr):
        if not hasattr(_stream, "reconfigure"):
            continue
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (ValueError, OSError):
            pass  # Best effort: some streams may not allow reconfiguration.


_ensure_utf8_stdout()

EPILOG = (
    "Privacy: traces are anonymized on fetch by default. "
    "Use --keep-raw to also retain a local raw copy (never shared)."
)

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("self-improve-cli")
except Exception:  # noqa: BLE001
    __version__ = "0.0.0"


def _load_env() -> None:
    """Load .env from cwd if it exists."""
    load_dotenv(Path(".env"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self-improve",
        description="Privacy-first CLI for inspecting AI-agent traces.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    discovery.register(sub)
    views.register(sub)
    metrics.register(sub)
    records.register(sub)
    system.register(sub)
    prompts.register(sub)
    ati.register(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env()
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    # keep_raw is only valid for fetch
    if not hasattr(args, "keep_raw"):
        args.keep_raw = False

    # --project overrides LANGSMITH_PROJECT (.env default), but is still
    # checked against the optional LANGSMITH_ALLOWED_PROJECTS allowlist.
    # Neither .env nor --project is a security control — security is enforced
    # by the API key's workspace scoping on the LangSmith server side.

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
