"""Metrics and verification commands — tool-metrics, context-metrics,
skill-metrics, compare, skill-check.
"""

from __future__ import annotations

import argparse
import json

from self_improve_cli.cli.common import (
    EXIT_ERROR,
    EXIT_OK,
    _get_store,
    _output,
    add_common_opts,
)


def _cmd_tool_metrics(args: argparse.Namespace) -> int:
    from self_improve_cli.metrics.tool_metrics import build_tool_metrics, tool_metrics_data

    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    if args.format == "json":
        _output(tool_metrics_data(trace.runs), args)
    else:
        content = build_tool_metrics(trace.runs)
        _output(content, args)
    return EXIT_OK


def _cmd_context_metrics(args: argparse.Namespace) -> int:
    from self_improve_cli.metrics.context_metrics import (
        build_context_metrics,
        context_metrics_data,
    )

    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    if args.format == "json":
        _output(context_metrics_data(trace.runs), args)
    else:
        content = build_context_metrics(trace.runs)
        _output(content, args)
    return EXIT_OK


def _cmd_skill_metrics(args: argparse.Namespace) -> int:
    from self_improve_cli.metrics.skill_metrics import (
        build_skill_metrics,
        skill_metrics_data,
    )

    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    if args.format == "json":
        _output(skill_metrics_data(trace.runs), args)
    else:
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
    else:
        match = expected in observed

    false_positive: str | None = None
    if not match and expected != "none" and observed:
        false_positive = f"{', '.join(observed)} triggered instead of {expected}"
    elif not match and expected == "none":
        false_positive = "skill triggered when none was expected"

    if args.format == "json":
        result = {
            "trace_id": args.trace_id,
            "expected": expected,
            "observed": observed,
            "match": match,
            "false_positive": false_positive,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        observed_str = ", ".join(observed) if observed else "none"
        status = "✅" if match else "❌"
        lines = [
            f"Expected: {expected}",
            f"Observed: {observed_str}  {status}",
        ]
        if false_positive:
            lines.append(f"(false positive: {false_positive})")
        print("\n".join(lines))
    return EXIT_OK if match else EXIT_ERROR


def register(sub: argparse._SubParsersAction) -> None:
    """Register metrics commands on the given subparsers action."""
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
    p = sub.add_parser("compare", help="Comparison: diff two traces (tokens, latency, skills)")
    p.add_argument("trace_a")
    p.add_argument("trace_b")
    add_common_opts(p)
    p.set_defaults(func=_cmd_compare)

    # skill-check
    p = sub.add_parser(
        "skill-check",
        help="Verification: check if the expected skill was triggered (exit 0=match, 1=mismatch)",
    )
    p.add_argument("trace_id")
    p.add_argument(
        "--expected",
        required=True,
        help="Expected skill name, or 'none' for anti-trigger check",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_skill_check)
