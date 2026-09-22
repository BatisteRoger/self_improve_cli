"""Trace-view commands — skeleton, narrative, run-detail, context-at,
target-timeline, error-neighborhood, tools.

All views are offline: they load a locally stored sanitized trace and render
a bounded representation. No network, no LLM calls.
"""

from __future__ import annotations

import argparse
import sys

from self_improve_cli.cli.common import (
    EXIT_OK,
    EXIT_USAGE,
    _get_store,
    _output,
    add_common_opts,
)
from self_improve_cli.representations import (
    build_narrative,
    build_skeleton,
    build_tools_detail,
    build_tools_overview,
    context_at,
    context_at_data,
    error_neighborhood,
    error_neighborhood_data,
    narrative_data,
    run_detail,
    run_detail_data,
    skeleton_data,
    target_timeline,
    target_timeline_data,
)


def _cmd_skeleton(args: argparse.Namespace) -> int:
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    errors_only = getattr(args, "errors_only", False)
    assessment = store.load_assessment(args.trace_id)
    if args.format == "json":
        data = skeleton_data(trace.runs, errors_only=errors_only)
        if assessment:
            data["assessment"] = {
                "task": assessment.task,
                "outcome": assessment.outcome.value,
                "outcome_source": assessment.outcome_source.value,
                "notes": assessment.notes,
                "assessed_at": assessment.assessed_at,
            }
        _output(data, args)
    else:
        content = build_skeleton(trace.runs, errors_only=errors_only)
        if assessment:
            header = (
                f"**Assessment: {assessment.outcome.value} "
                f"({assessment.outcome_source.value})** — {assessment.task}\n\n"
            )
            content = header + content
        _output(content, args)
    return EXIT_OK


def _cmd_narrative(args: argparse.Namespace) -> int:
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    mode = "full" if getattr(args, "full", False) else "compact"

    step_from = getattr(args, "from_step", None)
    step_to = getattr(args, "to_step", None)
    around = getattr(args, "around_step", None)
    if around is not None:
        step_from = around - 1
        step_to = around + 1

    if args.format == "json":
        _output(
            narrative_data(trace.runs, mode=mode, step_from=step_from, step_to=step_to),
            args,
        )
    else:
        content = build_narrative(trace.runs, mode=mode, step_from=step_from, step_to=step_to)
        _output(content, args)
    return EXIT_OK


def _cmd_run_detail(args: argparse.Namespace) -> int:
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    tool_calls_only = getattr(args, "tool_calls_only", False)
    inputs_only = getattr(args, "inputs_only", False)
    outputs_only = getattr(args, "outputs_only", False)

    # Enforce mutual exclusion (Recoverable contract): at most one of these
    # scoping flags may be active. Two together produce an empty or confusing
    # body silently — reject explicitly instead.
    active = [
        f
        for f, v in (
            ("--tool-calls-only", tool_calls_only),
            ("--inputs-only", inputs_only),
            ("--outputs-only", outputs_only),
        )
        if v
    ]
    if len(active) > 1:
        print(
            f"Error: {' and '.join(active)} are mutually exclusive. Pick one.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if args.format == "json":
        _output(
            run_detail_data(
                trace.runs,
                args.run_id,
                tool_calls_only=tool_calls_only,
                inputs_only=inputs_only,
                outputs_only=outputs_only,
            ),
            args,
        )
    else:
        content = run_detail(
            trace.runs,
            args.run_id,
            tool_calls_only=tool_calls_only,
            inputs_only=inputs_only,
            outputs_only=outputs_only,
        )
        _output(content, args)
    return EXIT_OK


def _cmd_context_at(args: argparse.Namespace) -> int:
    """Show what the model saw at a given main-loop step (bounded, selective)."""
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    from_step = getattr(args, "from_step", None)
    to_step = getattr(args, "to_step", None)
    tool_call_id = getattr(args, "tool_call_id", None)
    inputs_only = getattr(args, "inputs_only", False)
    outputs_only = getattr(args, "outputs_only", False)
    full = getattr(args, "full", False)

    # Enforce --from/--to pairing: both or neither (Recoverable contract).
    if (from_step is None) != (to_step is None):
        missing = "--to" if from_step is not None else "--from"
        print(
            f"Error: --from and --to must be used together. Missing {missing}.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # Enforce --inputs-only/--outputs-only mutual exclusion (Recoverable contract).
    if inputs_only and outputs_only:
        print(
            "Error: --inputs-only and --outputs-only are mutually exclusive. Pick one.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if args.format == "json":
        data = context_at_data(
            trace.runs,
            args.step,
            from_step=from_step,
            to_step=to_step,
            inputs_only=inputs_only,
            outputs_only=outputs_only,
            tool_call_id=tool_call_id,
            full=full,
        )
        _output(data, args)
    else:
        content = context_at(
            trace.runs,
            args.step,
            from_step=from_step,
            to_step=to_step,
            inputs_only=inputs_only,
            outputs_only=outputs_only,
            tool_call_id=tool_call_id,
            full=full,
        )
        _output(content, args)
    return EXIT_OK


def _cmd_target_timeline(args: argparse.Namespace) -> int:
    """Every step that touched a given target (file/path/key), in order."""
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    compact = getattr(args, "compact", False)
    if args.format == "json":
        _output(target_timeline_data(trace.runs, args.target, compact=compact), args)
    else:
        _output(target_timeline(trace.runs, args.target, compact=compact), args)
    return EXIT_OK


def _cmd_error_neighborhood(args: argparse.Namespace) -> int:
    """Steps around each error, with the agent's reaction."""
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    window = getattr(args, "window", 1)
    if args.format == "json":
        _output(error_neighborhood_data(trace.runs, window=window), args)
    else:
        _output(error_neighborhood(trace.runs, window=window), args)
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


def register(sub: argparse._SubParsersAction) -> None:
    """Register trace-view commands on the given subparsers action."""
    # skeleton
    p = sub.add_parser("skeleton", help="L1: one line per significant run")
    p.add_argument("trace_id")
    p.add_argument(
        "--errors-only",
        action="store_true",
        help="Show only runs with error or cancelled status (quick triage).",
    )
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
    p.add_argument(
        "--from",
        dest="from_step",
        type=int,
        metavar="N",
        help="Only show steps >= N (1-based narrative step numbers).",
    )
    p.add_argument(
        "--to",
        dest="to_step",
        type=int,
        metavar="N",
        help="Only show steps <= N (1-based narrative step numbers).",
    )
    p.add_argument(
        "--around-step",
        dest="around_step",
        type=int,
        metavar="N",
        help="Show steps N-1 to N+1 (convenience for --from/--to).",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_narrative)

    # run-detail
    p = sub.add_parser("run-detail", help="L3: full prompts/outputs of one run")
    p.add_argument("trace_id")
    p.add_argument("run_id")
    p.add_argument(
        "--tool-calls-only",
        action="store_true",
        help="Show only tool calls from the model output (LLM runs). "
        "Mutually exclusive with --inputs-only and --outputs-only.",
    )
    p.add_argument(
        "--inputs-only",
        action="store_true",
        help="Show only the input section (no output). Mirrors context-at --inputs-only. "
        "Mutually exclusive with --tool-calls-only and --outputs-only.",
    )
    p.add_argument(
        "--outputs-only",
        action="store_true",
        help="Show only the output section (no input). Mirrors context-at --outputs-only. "
        "Mutually exclusive with --tool-calls-only and --inputs-only.",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_run_detail)

    # context-at
    p = sub.add_parser(
        "context-at",
        help="L3: what the model saw at step N (bounded, selective)",
    )
    p.add_argument("trace_id")
    p.add_argument("step", type=int, help="0-based main-loop LLM step index")
    p.add_argument(
        "--from",
        dest="from_step",
        type=int,
        default=None,
        help="Diff mode: starting step (requires --to)",
    )
    p.add_argument(
        "--to",
        dest="to_step",
        type=int,
        default=None,
        help="Diff mode: ending step (requires --from)",
    )
    p.add_argument(
        "--inputs-only",
        action="store_true",
        help="Show only input messages (no output). Mutually exclusive with --outputs-only.",
    )
    p.add_argument(
        "--outputs-only",
        action="store_true",
        help="Show only the model output (no input messages). "
        "Mutually exclusive with --inputs-only.",
    )
    p.add_argument(
        "--tool",
        dest="tool_call_id",
        default=None,
        help="Show only the tool result matching this tool_call_id",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Do not truncate message text (default: bounded preview)",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_context_at)

    # target-timeline
    p = sub.add_parser(
        "target-timeline",
        help="Navigation: every step that touched a target (file/path/key)",
    )
    p.add_argument("trace_id")
    p.add_argument("target", help="Target string (file path, key, tool name)")
    p.add_argument(
        "--compact",
        action="store_true",
        help="Show only tool name + status per step (no args/result). "
        "Useful for overview questions on long traces.",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_target_timeline)

    # error-neighborhood
    p = sub.add_parser(
        "error-neighborhood",
        help="Navigation: steps around each error, with the agent's reaction",
    )
    p.add_argument("trace_id")
    p.add_argument(
        "--window",
        type=int,
        default=1,
        help="Number of steps to show before and after each error (default: 1)",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_error_neighborhood)

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
