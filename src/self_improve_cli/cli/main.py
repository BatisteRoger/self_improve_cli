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
from self_improve_cli.representations import (
    build_narrative,
    build_skeleton,
    build_tools_detail,
    build_tools_overview,
    run_detail,
    write_ter,
)
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
    trace, report = anonymize_trace(trace)

    sanitized_path = store.save_sanitized(trace)

    stats = write_ter(trace_id, trace.runs, store)

    result: dict[str, Any] = {
        "trace_id": trace_id,
        "runs": len(trace.runs),
        "sanitized": True,
        "sanitization_report": report.to_dict(),
        "presidio_available": is_presidio_available(),
        "sanitized_path": str(sanitized_path),
        "raw_saved": args.keep_raw,
        "ter_stats": stats,
    }
    if getattr(args, "from_run", False):
        result["resolved_from_run"] = args.trace_id
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
    mode = "full" if getattr(args, "full", False) else "compact"
    content = build_narrative(trace.runs, mode=mode)
    _output(content, args)
    return EXIT_OK


def _cmd_tool_metrics(args: argparse.Namespace) -> int:
    from self_improve_cli.metrics.tool_metrics import build_tool_metrics

    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    content = build_tool_metrics(trace.runs)
    _output(content, args)
    return EXIT_OK


def _cmd_skill_metrics(args: argparse.Namespace) -> int:
    from self_improve_cli.metrics.skill_metrics import build_skill_metrics

    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    content = build_skill_metrics(trace.runs)
    _output(content, args)
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare two traces: tokens, latency, tool calls, and skill invocations."""
    from self_improve_cli.metrics.skill_metrics import skill_invocations
    from self_improve_cli.metrics.tool_metrics import tool_call_counts

    store = _get_store(args)
    trace_a = store.load_trace(args.trace_a)
    trace_b = store.load_trace(args.trace_b)

    from self_improve_cli.representations import significant_runs

    sig_a = significant_runs(trace_a.runs)
    sig_b = significant_runs(trace_b.runs)

    def _total_tokens(sig: list) -> int:
        return sum(r.total_tokens or 0 for r in sig if r.run_type.value == "llm")

    def _llm_count(sig: list) -> int:
        return sum(1 for r in sig if r.run_type.value == "llm")

    def _latency_ms(sig: list) -> int | None:
        if not sig or not sig[0].start_time:
            return None
        from datetime import datetime

        try:
            start = datetime.fromisoformat(sig[0].start_time.replace("Z", "+00:00"))
            end_run = max(sig, key=lambda r: r.end_time or r.start_time or "")
            if not end_run.end_time:
                return None
            end = datetime.fromisoformat(end_run.end_time.replace("Z", "+00:00"))
            return int((end - start).total_seconds() * 1000)
        except (ValueError, TypeError):
            return None

    tokens_a = _total_tokens(sig_a)
    tokens_b = _total_tokens(sig_b)
    llm_a = _llm_count(sig_a)
    llm_b = _llm_count(sig_b)
    latency_a = _latency_ms(sig_a)
    latency_b = _latency_ms(sig_b)
    tools_a = tool_call_counts(trace_a.runs)
    tools_b = tool_call_counts(trace_b.runs)
    skills_a = skill_invocations(trace_a.runs)
    skills_b = skill_invocations(trace_b.runs)

    skill_names_a = [s["skill_name"] for s in skills_a if s["skill_name"]]
    skill_names_b = [s["skill_name"] for s in skills_b if s["skill_name"]]

    if args.format == "json":
        result = {
            "trace_a": args.trace_a,
            "trace_b": args.trace_b,
            "token_delta": tokens_b - tokens_a,
            "latency_delta_ms": (latency_b - latency_a) if latency_a and latency_b else None,
            "llm_calls_a": llm_a,
            "llm_calls_b": llm_b,
            "tool_calls_a": tools_a,
            "tool_calls_b": tools_b,
            "skills_a": skill_names_a,
            "skills_b": skill_names_b,
        }
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    else:
        lines = [
            "# Trace comparison",
            "",
            f"| Metric | Trace A ({args.trace_a[:8]}) | Trace B ({args.trace_b[:8]}) | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| Total tokens | {tokens_a:,} | {tokens_b:,} | {tokens_b - tokens_a:+,} |",
            f"| LLM calls | {llm_a} | {llm_b} | {llm_b - llm_a:+d} |",
            f"| Latency | {latency_a}ms | {latency_b}ms | "
            + (f"{(latency_b - latency_a):+d}ms" if latency_a and latency_b else "N/A")
            + " |",
            "",
            "Skills triggered:",
            f"- A: {', '.join(skill_names_a) or '(none)'}",
            f"- B: {', '.join(skill_names_b) or '(none)'}",
        ]
        print("\n".join(lines))
    return EXIT_OK


def _cmd_skill_check(args: argparse.Namespace) -> int:
    """Check if the expected skill was triggered in a trace."""
    from self_improve_cli.metrics.skill_metrics import skill_invocations

    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    invocations = skill_invocations(trace.runs)
    observed = [s["skill_name"] for s in invocations if s["skill_name"]]
    expected = args.expected

    if expected == "none":
        match = len(observed) == 0
        observed_str = ", ".join(observed) if observed else "none"
    else:
        match = expected in observed
        observed_str = ", ".join(observed) if observed else "none"

    status = "✅" if match else "❌"
    lines = [
        f"Expected: {expected}",
        f"Observed: {observed_str}  {status}",
    ]
    if not match and expected != "none" and observed:
        lines.append(f"(false positive: {observed_str} triggered instead of {expected})")
    elif not match and expected == "none":
        lines.append("(false positive: skill triggered when none was expected)")

    print("\n".join(lines))
    return EXIT_OK if match else EXIT_ERROR


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


def _cmd_tools(args: argparse.Namespace) -> int:
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    if getattr(args, "detail", False):
        content = build_tools_detail(trace.runs, tool_name=getattr(args, "tool_name", None))
    else:
        content = build_tools_overview(trace.runs)
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


def _find_skills_dir() -> Path | None:
    """Find the skills directory, checking repo root, .agents, and package data."""
    # 1. Repo root (for cloned repo users)
    root_skills = Path("skills")
    if root_skills.is_dir() and any(root_skills.glob("*/SKILL.md")):
        return root_skills
    # 2. .agents/skills (for users who installed via npx skills add)
    agents_skills = Path(".agents/skills")
    if agents_skills.is_dir() and any(agents_skills.glob("*/SKILL.md")):
        return agents_skills
    # 3. Package data (for pip-installed users)
    try:
        from importlib.resources import files

        pkg_skills = Path(str(files("self_improve_cli"))) / "skills"
        if pkg_skills.is_dir() and any(pkg_skills.glob("*/SKILL.md")):
            return pkg_skills
    except Exception:  # noqa: BLE001
        pass
    return None


def _cmd_skill(args: argparse.Namespace) -> int:
    """List available skills or print a specific skill's SKILL.md."""
    skills_dir = _find_skills_dir()
    if skills_dir is None:
        print(
            "No skills found. Install with: npx skills add BatisteRoger/self_improve_cli",
            file=sys.stderr,
        )
        return EXIT_ERROR

    skill_name = getattr(args, "skill_name", None)
    if skill_name is None:
        # List available skills
        skills = sorted(
            d.name for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
        )
        if not skills:
            print("No skills found.", file=sys.stderr)
            return EXIT_ERROR
        for name in skills:
            print(name)
        return EXIT_OK

    # Print a specific skill
    skill_path = skills_dir / skill_name / "SKILL.md"
    if not skill_path.exists():
        available = sorted(
            d.name for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
        )
        print(
            f"Skill '{skill_name}' not found. Available: {', '.join(available)}",
            file=sys.stderr,
        )
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


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Run a read-only local setup diagnostic."""
    from self_improve_cli.cli.doctor import format_report_markdown, run_doctor

    report = run_doctor(profile=args.profile, offline=args.offline)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report_markdown(report))
    return EXIT_OK if not report.has_failures else EXIT_ERROR


# ---------------------------------------------------------------------------
# Prompt commands
# ---------------------------------------------------------------------------


def _cmd_prompt_pull(args: argparse.Namespace) -> int:
    """Pull a prompt from LangSmith Prompt Hub and save it locally."""
    if os.environ.get("ENABLE_PROMPT_HUB", "").lower() != "true":
        print(
            "Prompt Hub access is disabled. Set ENABLE_PROMPT_HUB=true in .env to enable.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    from self_improve_cli.sources.langsmith import LangSmithSource

    source = LangSmithSource()
    tag = args.tag or "latest"
    try:
        content = source.pull_prompt(args.name, tag=tag, workspace_id=args.workspace)
    except Exception as e:
        print(
            f"Failed to pull prompt '{args.name}:{tag}'"
            f"{' from workspace ' + args.workspace if args.workspace else ''}.\n"
            f"Error: {e}\n"
            f"Tip: try without --tag to pull the latest version.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    store = _get_store(args)
    path = store.save_prompt(args.name, tag, content)
    result = {
        "name": args.name,
        "tag": tag,
        "path": str(path),
        "size_bytes": len(content.encode("utf-8")),
    }
    _output(result, args)
    return EXIT_OK


def _cmd_prompt_list(args: argparse.Namespace) -> int:
    """List locally saved prompts."""
    store = _get_store(args)
    prompts = store.list_prompts()
    if args.format == "json":
        print(json.dumps(prompts, indent=2, ensure_ascii=False))
    else:
        if not prompts:
            print("No saved prompts. Use `self-improve prompt pull <name>` to download one.")
            return EXIT_OK
        for p in prompts:
            print(f"{p['name']}:{p['tag']}  {p['size_bytes']} bytes")
    return EXIT_OK


def _cmd_prompt_show(args: argparse.Namespace) -> int:
    """Show a locally saved prompt."""
    store = _get_store(args)
    tag = args.tag or "latest"
    content = store.load_prompt(args.name, tag=tag)
    if content is None:
        print(
            f"Prompt '{args.name}:{tag}' not found. "
            f"Pull it first: self-improve prompt pull {args.name} --tag {tag}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    _output(content, args)
    return EXIT_OK


def _cmd_prompt_diff(args: argparse.Namespace) -> int:
    """Compare a locally saved prompt against what a trace used (approximate)."""
    import difflib

    store = _get_store(args)
    tag = args.tag or "latest"
    local_prompt = store.load_prompt(args.name, tag=tag)
    if local_prompt is None:
        print(f"Prompt '{args.name}:{tag}' not found. Pull it first.", file=sys.stderr)
        return EXIT_ERROR

    trace = store.load_trace(args.trace_id)
    # Extract the system message from the first LLM run that has one
    trace_prompt = ""
    for run in trace.runs:
        if run.run_type.value != "llm":
            continue
        for msg in run.input_messages:
            if msg.role == "system":
                trace_prompt = msg.text
                break
        if trace_prompt:
            break

    if not trace_prompt:
        print(f"No system message found in trace {args.trace_id}.", file=sys.stderr)
        return EXIT_ERROR

    local_lines = local_prompt.splitlines(keepends=True)
    trace_lines = trace_prompt.splitlines(keepends=True)
    diff = difflib.unified_diff(
        local_lines,
        trace_lines,
        fromfile=f"{args.name}:{tag} (local)",
        tofile=f"{args.trace_id} (trace)",
    )
    diff_text = "".join(diff)
    if not diff_text:
        print(f"No differences found between {args.name}:{tag} and the trace's system prompt.")
    else:
        print("Approximate diff (traces may have runtime substitutions):\n")
        print(diff_text, end="")
    return EXIT_OK


# ---------------------------------------------------------------------------
# ATI commands
# ---------------------------------------------------------------------------


def _cmd_ati_list(args: argparse.Namespace) -> int:
    """List registered ATIs."""
    store = _get_store(args)
    atis = store.list_atis()
    if args.format == "json":
        print(json.dumps(atis, ensure_ascii=False))
    else:
        if not atis:
            print("No ATIs registered. Use the document-ati skill to create one.")
            return EXIT_OK
        for name in atis:
            print(name)
    return EXIT_OK


def _cmd_ati_show(args: argparse.Namespace) -> int:
    """Show an ATI's architecture document."""
    store = _get_store(args)
    content = store.load_ati(args.name)
    if content is None:
        print(
            f"ATI '{args.name}' not found. Create it with the document-ati skill.", file=sys.stderr
        )
        return EXIT_ERROR
    _output(content, args)
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
        "parent trace ID before fetching. Useful when you only have a run ID "
        "(e.g. from a LangSmith trace URL).",
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
    p.add_argument(
        "--full",
        action="store_true",
        help="Use full narrative (common-prefix diff). Default: compact (per-index diff).",
    )
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

    # skill-metrics
    p = sub.add_parser("skill-metrics", help="L1: skill invocations & token cost")
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_skill_metrics)

    # compare
    p = sub.add_parser("compare", help="Compare two traces: tokens, latency, skills")
    p.add_argument("trace_a")
    p.add_argument("trace_b")
    add_common_opts(p)
    p.set_defaults(func=_cmd_compare)

    # skill-check
    p = sub.add_parser(
        "skill-check",
        help="Check if the expected skill was triggered (exit 0=match, 1=mismatch)",
    )
    p.add_argument("trace_id")
    p.add_argument(
        "--expected",
        required=True,
        help="Expected skill name, or 'none' for anti-trigger check",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_skill_check)

    # run-detail
    p = sub.add_parser("run-detail", help="L3: full prompts/outputs of one run")
    p.add_argument("trace_id")
    p.add_argument("run_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_run_detail)

    # tools
    p = sub.add_parser(
        "tools",
        help="L1: tools available per LLM run (matrix overview, or --detail for docstrings)",
    )
    p.add_argument("trace_id")
    p.add_argument(
        "--detail",
        action="store_true",
        help="Show full docstrings and parameter schemas instead of the matrix.",
    )
    p.add_argument(
        "tool_name",
        nargs="?",
        default=None,
        help="With --detail: show only this tool (omit for all tools).",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_tools)

    # info
    p = sub.add_parser("info", help="Show info about a saved trace")
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_info)

    # skill
    p = sub.add_parser("skill", help="List available skills or print a specific skill")
    p.add_argument("skill_name", nargs="?", default=None, help="Skill name (omit to list)")
    p.set_defaults(func=_cmd_skill)

    # init
    p = sub.add_parser("init", help="Create a .env file from .env.example")
    p.set_defaults(func=_cmd_init)

    # doctor
    p = sub.add_parser(
        "doctor",
        help="Read-only local setup diagnostic (no installs, no network)",
    )
    p.add_argument(
        "--profile",
        choices=["default", "fetch"],
        default="default",
        help="Check profile: 'default' for analysis commands, 'fetch' escalates "
        "langsmith_extra and env_api_key to fail (default: default)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Skip checks requiring Docker, databases, or network (no-op for this "
        "project — all checks are already local-only)",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_doctor)

    # prompt
    p = sub.add_parser("prompt", help="Pull and manage LangSmith prompts (read-only)")
    prompt_sub = p.add_subparsers(dest="prompt_command", required=True)

    p_pull = prompt_sub.add_parser("pull", help="Download a prompt from LangSmith Prompt Hub")
    p_pull.add_argument("name", help="Prompt name (e.g. react_agent)")
    p_pull.add_argument(
        "--tag", default=None, help="Tag (e.g. prod, staging, test). Default: latest"
    )
    p_pull.add_argument("--project", default=None, help="LangSmith project name")
    p_pull.add_argument(
        "--workspace",
        default=None,
        help="LangSmith workspace ID for non-default workspaces (e.g. Workspace 2, Workspace 3)",
    )
    add_common_opts(p_pull)
    p_pull.set_defaults(func=_cmd_prompt_pull)

    p_list = prompt_sub.add_parser("list", help="List locally saved prompts")
    add_common_opts(p_list)
    p_list.set_defaults(func=_cmd_prompt_list)

    p_show = prompt_sub.add_parser("show", help="Show a locally saved prompt")
    p_show.add_argument("name", help="Prompt name")
    p_show.add_argument("--tag", default=None, help="Tag (default: latest)")
    add_common_opts(p_show)
    p_show.set_defaults(func=_cmd_prompt_show)

    p_diff = prompt_sub.add_parser(
        "diff", help="Compare a saved prompt against what a trace used (approximate)"
    )
    p_diff.add_argument("name", help="Prompt name")
    p_diff.add_argument("trace_id", help="Trace ID to compare against")
    p_diff.add_argument("--tag", default=None, help="Tag (default: latest)")
    add_common_opts(p_diff)
    p_diff.set_defaults(func=_cmd_prompt_diff)

    # ati
    p = sub.add_parser("ati", help="List or show target agent architecture docs")
    ati_sub = p.add_subparsers(dest="ati_command", required=True)

    p_ati_list = ati_sub.add_parser("list", help="List registered ATIs")
    add_common_opts(p_ati_list)
    p_ati_list.set_defaults(func=_cmd_ati_list)

    p_ati_show = ati_sub.add_parser("show", help="Show an ATI's architecture document")
    p_ati_show.add_argument("name", help="ATI name")
    add_common_opts(p_ati_show)
    p_ati_show.set_defaults(func=_cmd_ati_show)

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
