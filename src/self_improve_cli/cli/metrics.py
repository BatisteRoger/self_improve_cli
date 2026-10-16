"""Metrics and verification commands — tool-metrics, context-metrics,
skill-metrics, compare, gate, skill-check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from self_improve_cli.cli.common import (
    EXIT_ERROR,
    EXIT_GATE_FAIL,
    EXIT_GATE_INCONCLUSIVE,
    EXIT_GATE_PASS,
    EXIT_OK,
    EXIT_USAGE,
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
    token_delta = tokens_b - tokens_a if tokens_a is not None and tokens_b is not None else None
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
            "token_delta": token_delta,
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

        def _tok(n: int | None) -> str:
            return "n/a" if n is None else f"{n:,}"

        lines = [
            "# Trace comparison",
            "",
            f"| Metric | Trace A ({args.trace_a[:8]}) | Trace B ({args.trace_b[:8]}) | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| Total tokens | {_tok(tokens_a)} | {_tok(tokens_b)} | "
            + ("n/a" if token_delta is None else f"{token_delta:+,}")
            + " |",
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


_GATE_EXIT = {
    "pass": EXIT_GATE_PASS,
    "fail": EXIT_GATE_FAIL,
    "inconclusive": EXIT_GATE_INCONCLUSIVE,
}


def _models(trace: Any) -> set[str]:
    """Model ids from the LangChain `ls_model_name` run metadata (LLM runs)."""
    found: set[str] = set()
    for run in trace.runs:
        if run.run_type.value != "llm":
            continue
        meta = run.extra.get("metadata") or (run.extra.get("extra") or {}).get("metadata") or {}
        name = meta.get("ls_model_name") if isinstance(meta, dict) else None
        if name:
            found.add(str(name))
    return found


def _root_run(trace: Any) -> Any:
    return next((r for r in trace.runs if r.parent_run_id is None), None)


def _parse_id_file(path: str, store: Any) -> list[tuple[str, str]]:
    """Read `trace_id` or `example_id trace_id` lines into (example_id, trace_id) pairs."""
    pairs: list[tuple[str, str]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, 1):
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
            continue
        if len(parts) != 1:
            raise ValueError(
                f"{path}:{lineno}: expected `trace_id` or `example_id trace_id`, got {line!r}."
            )
        root = _root_run(store.load_trace(parts[0]))
        example = root.extra.get("reference_example_id") if root else None
        if not example:
            raise ValueError(
                f"{path}:{lineno}: trace {parts[0]} has no reference_example_id. "
                "Write the line as `example_id trace_id`."
            )
        pairs.append((str(example), parts[0]))
    if not pairs:
        raise ValueError(f"{path}: no trace IDs found.")
    return pairs


def _load_side(arg: str, label: str, store: Any, override: Any) -> Any:
    """Resolve `@file` or an experiment name/session ID to a gate `Side` (offline)."""
    from self_improve_cli.domain import OutcomePolicy, OutcomeStatus, outcome_from_policy
    from self_improve_cli.metrics.efficiency import efficiency_summary
    from self_improve_cli.metrics.gate_sets import RunOutcome, Side

    binary = {OutcomeStatus.SUCCESS.value: "pass", OutcomeStatus.FAIL.value: "fail"}

    def _from_feedback(trace_id: str) -> str:
        status = outcome_from_policy(store.load_feedback(trace_id), override)[0]
        return binary.get(status.value, "unknown")

    groups: dict[str, list[Any]] = {}
    models: set[str] = set()
    starts: list[str] = []

    def _add(example_id: str, trace_id: str, outcome: str) -> None:
        trace = store.load_trace(trace_id)
        models.update(_models(trace))
        root = _root_run(trace)
        if root is not None and root.start_time:
            starts.append(root.start_time)
        groups.setdefault(example_id, []).append(
            RunOutcome(trace_id, outcome, efficiency_summary(trace.runs))
        )

    if arg.startswith("@"):
        for example_id, trace_id in _parse_id_file(arg[1:], store):
            if override is not None:
                outcome = _from_feedback(trace_id)
            else:
                assessment = store.load_assessment(trace_id)
                outcome = (
                    binary.get(assessment.outcome.value, "unknown") if assessment else "unknown"
                )
            _add(example_id, trace_id, outcome)
        policy = override or OutcomePolicy(None, None, "assessment")
        conditions: dict[str, Any] = {
            "start_time": min(starts) if starts else None,
            "end_time": max(starts) if starts else None,
            "models": sorted(models) or None,
        }
        return Side(label, "file", arg[1:], policy, groups, conditions, None)

    receipt = store.find_experiment(arg)
    if receipt is None:
        raise ValueError(
            f"No stored experiment named or with session ID '{arg}'. "
            f"Fetch it first: `self-improve fetch --experiment '{arg}'`, or pass `@file`."
        )
    for example_id, runs in receipt.get("examples", {}).items():
        for run in runs:
            if override is not None:
                outcome = _from_feedback(run["trace_id"])
            else:
                outcome = binary.get(run.get("outcome", "unknown"), "unknown")
            _add(example_id, run["trace_id"], outcome)
    policy = override or OutcomePolicy.from_dict(receipt.get("outcome_policy"))
    conditions = {**(receipt.get("conditions") or {}), "models": sorted(models) or None}
    return Side(
        receipt.get("experiment_name") or label,
        "experiment",
        receipt["session_id"],
        policy,
        groups,
        conditions,
        receipt.get("dataset_id"),
    )


def _cmd_gate_sets(args: argparse.Namespace) -> int:
    """Experiment-set verdict: per-example flips above the baseline noise floor."""
    from self_improve_cli.domain import parse_outcome_key
    from self_improve_cli.metrics.gate_sets import build_gate_sets, gate_sets_data

    override = parse_outcome_key(args.outcome_key, source="gate") if args.outcome_key else None
    store = _get_store(args)
    baseline = _load_side(args.baseline, "baseline", store, override)
    candidate = _load_side(args.candidate, "candidate", store, override)
    if baseline.kind == candidate.kind and baseline.ref == candidate.ref:
        raise ValueError("--baseline and --candidate resolve to the same set.")
    if baseline.dataset_id and candidate.dataset_id and baseline.dataset_id != candidate.dataset_id:
        raise ValueError(
            f"Experiments use different datasets ({baseline.dataset_id} vs "
            f"{candidate.dataset_id}); examples cannot be paired."
        )
    data = gate_sets_data(baseline, candidate, strict_conditions=args.strict_conditions)
    _output(data if args.format == "json" else build_gate_sets(data), args)
    return _GATE_EXIT[data["verdict"]]


def _cmd_gate(args: argparse.Namespace) -> int:
    """Before/after verdict receipt: pass / fail / inconclusive, exit 0 / 1 / 2."""
    from self_improve_cli.metrics.efficiency import efficiency_summary
    from self_improve_cli.metrics.gate import Verdict, build_gate, gate_data

    sets_mode = args.baseline is not None or args.candidate is not None
    pair_mode = args.trace_before is not None or args.trace_after is not None
    usage = None
    if sets_mode and pair_mode:
        usage = "use either `gate <before> <after>` or `gate --baseline X --candidate Y`."
    elif sets_mode and (args.baseline is None or args.candidate is None):
        usage = "--baseline and --candidate must be given together."
    elif not sets_mode and (args.trace_before is None or args.trace_after is None):
        usage = "pass two trace IDs, or --baseline <exp|@file> --candidate <exp|@file>."
    elif not sets_mode and (args.outcome_key or args.strict_conditions):
        usage = "--outcome-key and --strict-conditions apply to --baseline/--candidate only."
    if usage:
        print(f"Error: {usage}", file=sys.stderr)
        return EXIT_USAGE
    if sets_mode:
        return _cmd_gate_sets(args)

    before_id, after_id = str(args.trace_before), str(args.trace_after)
    if before_id == after_id:
        raise ValueError(
            f"trace_before and trace_after are the same ({before_id}). "
            "Pass the trace recorded before the change, then the one after."
        )
    store = _get_store(args)
    before = store.load_trace(before_id)
    after = store.load_trace(after_id)
    data = gate_data(
        before_id,
        after_id,
        store.load_assessment(before_id),
        store.load_assessment(after_id),
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
    p.add_argument("trace_before", nargs="?", default=None)
    p.add_argument("trace_after", nargs="?", default=None)
    p.add_argument(
        "--baseline",
        default=None,
        help="Experiment set mode: stored experiment name/session ID, or @file of trace IDs",
    )
    p.add_argument("--candidate", default=None, help="Same forms as --baseline")
    p.add_argument(
        "--outcome-key",
        default=None,
        help="Set mode: re-derive both sides' outcomes from stored feedback ('KEY' or 'KEY>=T')",
    )
    p.add_argument(
        "--strict-conditions",
        action="store_true",
        help="Set mode: differing conditions (model, commit, dataset version, policy) "
        "make the verdict inconclusive (conditions_differ)",
    )
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
