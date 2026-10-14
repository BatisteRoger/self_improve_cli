"""Metrics and verification commands — tool-metrics, context-metrics,
skill-metrics, compare, gate, skill-check.
"""

from __future__ import annotations

import argparse
import json

from self_improve_cli.cli.common import (
    EXIT_ERROR,
    EXIT_GATE_FAIL,
    EXIT_GATE_INCONCLUSIVE,
    EXIT_GATE_PASS,
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
    from self_improve_cli.metrics.efficiency import efficiency_summary
    from self_improve_cli.metrics.skill_metrics import skill_invocations

    store = _get_store(args)
    trace_a = store.load_trace(args.trace_a)
    trace_b = store.load_trace(args.trace_b)

    eff_a = efficiency_summary(trace_a.runs)
    eff_b = efficiency_summary(trace_b.runs)
    tokens_a, tokens_b = eff_a.tokens, eff_b.tokens
    llm_a, llm_b = eff_a.llm_calls, eff_b.llm_calls
    latency_a, latency_b = eff_a.latency_ms, eff_b.latency_ms
    tools_a, tools_b = eff_a.tool_calls, eff_b.tool_calls
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


def _cmd_gate(args: argparse.Namespace) -> int:
    """Before/after verdict receipt: pass / fail / inconclusive, exit 0 / 1 / 2."""
    from self_improve_cli.metrics.efficiency import efficiency_summary
    from self_improve_cli.metrics.gate import Verdict, build_gate, gate_data

    if args.trace_before == args.trace_after:
        raise ValueError(
            f"trace_before and trace_after are the same ({args.trace_before}). "
            "Pass the trace recorded before the change, then the one after."
        )
    store = _get_store(args)
    before = store.load_trace(args.trace_before)
    after = store.load_trace(args.trace_after)
    data = gate_data(
        args.trace_before,
        args.trace_after,
        store.load_assessment(args.trace_before),
        store.load_assessment(args.trace_after),
        efficiency_summary(before.runs),
        efficiency_summary(after.runs),
    )
    _output(data if args.format == "json" else build_gate(data), args)
    return {
        Verdict.PASS: EXIT_GATE_PASS,
        Verdict.FAIL: EXIT_GATE_FAIL,
        Verdict.INCONCLUSIVE: EXIT_GATE_INCONCLUSIVE,
    }[Verdict(data["verdict"])]


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

    # gate
    p = sub.add_parser(
        "gate",
        help="Verification: before/after verdict (exit 0=pass, 1=fail, 2=inconclusive)",
    )
    p.add_argument("trace_before")
    p.add_argument("trace_after")
    add_common_opts(p)
    p.set_defaults(func=_cmd_gate)

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
