"""Prompt commands — pull, list, show, diff for LangSmith Prompt Hub."""

from __future__ import annotations

import argparse
import json
import os
import sys

from self_improve_cli.cli.common import (
    EXIT_ERROR,
    EXIT_OK,
    _get_store,
    _output,
    add_common_opts,
)


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
    if args.format == "json":
        result = {
            "prompt_name": args.name,
            "tag": tag,
            "trace_id": args.trace_id,
            "has_differences": bool(diff_text),
            "diff": diff_text,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif not diff_text:
        print(f"No differences found between {args.name}:{tag} and the trace's system prompt.")
    else:
        print("Approximate diff (traces may have runtime substitutions):\n")
        print(diff_text, end="")
    return EXIT_OK


def register(sub: argparse._SubParsersAction) -> None:
    """Register prompt commands on the given subparsers action."""
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
