"""Analyst-record commands — info, assess, finding.

These read and write analyst-authored metadata stored alongside the trace
(`assessment.json`, `findings.json`). See domain/records.py for the types.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from self_improve_cli.cli.common import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    _get_store,
    _output,
    add_common_opts,
)
from self_improve_cli.domain import Assessment, Finding, OutcomeSource, OutcomeStatus


def _cmd_info(args: argparse.Namespace) -> int:
    """Show info about a saved trace (sanitization status, stats)."""
    store = _get_store(args)
    trace = store.load_trace(args.trace_id)
    result: dict[str, Any] = {
        "trace_id": trace.trace_id,
        "sanitized": trace.sanitized,
        "sanitization_report": trace.sanitization_report,
        "source": trace.source,
        "schema_version": trace.schema_version,
        "run_count": len(trace.runs),
    }
    assessment = store.load_assessment(args.trace_id)
    if assessment:
        result["assessment"] = {
            "task": assessment.task,
            "outcome": assessment.outcome.value,
            "outcome_source": assessment.outcome_source.value,
            "notes": assessment.notes,
            "assessed_at": assessment.assessed_at,
        }
    findings = store.load_findings(args.trace_id)
    if findings:
        result["findings"] = {"count": len(findings), "ids": [f.id for f in findings]}
    _output(result, args)
    return EXIT_OK


def _finding_to_dict(f: Finding) -> dict[str, Any]:
    return {
        "id": f.id,
        "trace_id": f.trace_id,
        "title": f.title,
        "pattern": f.pattern,
        "secondary_patterns": f.secondary_patterns,
        "impact": f.impact,
        "evidence_strength": f.evidence_strength,
        "triangle_axis": f.triangle_axis,
        "evidence": f.evidence,
        "assessment": f.assessment,
        "fault_locus": f.fault_locus,
        "suggested_next_action": f.suggested_next_action,
        "candidate_improvement": f.candidate_improvement,
        "validation": f.validation,
        "created_at": f.created_at,
    }


def _cmd_finding(args: argparse.Namespace) -> int:
    """List, add, remove, or clear persisted findings for a trace."""
    store = _get_store(args)

    if args.finding_command == "add":
        from datetime import UTC, datetime

        finding = Finding(
            id=store.next_finding_id(args.trace_id),
            trace_id=args.trace_id,
            title=args.title,
            pattern=args.pattern,
            secondary_patterns=args.secondary_pattern or [],
            impact=args.impact,
            evidence_strength=args.strength,
            triangle_axis=args.axis,
            evidence=args.evidence or [],
            assessment=args.assessment or "",
            fault_locus=args.locus or "",
            suggested_next_action=args.next or "",
            candidate_improvement=args.candidate or "",
            validation=args.validation or "",
            created_at=datetime.now(UTC).isoformat(),
        )
        store.add_finding(finding)
        _output(_finding_to_dict(finding), args)
        return EXIT_OK

    if args.finding_command == "remove":
        if store.remove_finding(args.trace_id, args.finding_id):
            print(f"Removed finding {args.finding_id} from trace {args.trace_id}", file=sys.stderr)
        else:
            print(
                f"No finding {args.finding_id} for trace {args.trace_id}. "
                "Run `self-improve finding list <trace_id>` for valid IDs.",
                file=sys.stderr,
            )
            return EXIT_ERROR
        return EXIT_OK

    if args.finding_command == "clear":
        if store.clear_findings(args.trace_id):
            print(f"Cleared findings for trace {args.trace_id}", file=sys.stderr)
        else:
            print(f"No findings found for trace {args.trace_id}", file=sys.stderr)
        return EXIT_OK

    # list
    findings = store.load_findings(args.trace_id)
    if args.format == "json":
        print(
            json.dumps(
                {"trace_id": args.trace_id, "findings": [_finding_to_dict(f) for f in findings]},
                indent=2,
                default=str,
                ensure_ascii=False,
            )
        )
    else:
        if not findings:
            print(f"No findings for trace {args.trace_id}.")
        for f in findings:
            print(f"{f.id} | {f.pattern} | {f.impact} | {f.evidence_strength} | {f.title}")
    return EXIT_OK


def _cmd_assess(args: argparse.Namespace) -> int:
    """Set, show, or clear a manual task & outcome assessment for a trace."""
    store = _get_store(args)

    if args.clear:
        if store.clear_assessment(args.trace_id):
            print(f"Cleared assessment for trace {args.trace_id}", file=sys.stderr)
        else:
            print(f"No assessment found for trace {args.trace_id}", file=sys.stderr)
        return EXIT_OK

    # If no set-mode flags provided, show the current assessment.
    set_mode = args.task or args.outcome or args.source or args.notes is not None
    if not set_mode:
        assessment = store.load_assessment(args.trace_id)
        if assessment is None:
            print(
                f"No assessment set for trace {args.trace_id}. "
                "Use --task and --outcome to set one.",
                file=sys.stderr,
            )
            return EXIT_ERROR
        result = {
            "trace_id": assessment.trace_id,
            "task": assessment.task,
            "outcome": assessment.outcome.value,
            "outcome_source": assessment.outcome_source.value,
            "notes": assessment.notes,
            "assessed_at": assessment.assessed_at,
        }
        _output(result, args)
        return EXIT_OK

    # Set or update the assessment.
    existing = store.load_assessment(args.trace_id)
    task = args.task if args.task else (existing.task if existing else "")
    if not task:
        print("Error: --task is required when setting an assessment.", file=sys.stderr)
        return EXIT_USAGE

    outcome = (
        OutcomeStatus(args.outcome)
        if args.outcome
        else (existing.outcome if existing else OutcomeStatus.UNKNOWN)
    )
    source = (
        OutcomeSource(args.source)
        if args.source
        else (existing.outcome_source if existing else OutcomeSource.UNKNOWN)
    )
    notes = args.notes if args.notes is not None else (existing.notes if existing else "")

    from datetime import UTC, datetime

    assessment = Assessment(
        trace_id=args.trace_id,
        task=task,
        outcome=outcome,
        outcome_source=source,
        notes=notes,
        assessed_at=datetime.now(UTC).isoformat(),
    )
    store.save_assessment(assessment)
    result = {
        "trace_id": assessment.trace_id,
        "task": assessment.task,
        "outcome": assessment.outcome.value,
        "outcome_source": assessment.outcome_source.value,
        "notes": assessment.notes,
        "assessed_at": assessment.assessed_at,
    }
    _output(result, args)
    return EXIT_OK


def register(sub: argparse._SubParsersAction) -> None:
    """Register analyst-record commands on the given subparsers action."""
    # info
    p = sub.add_parser("info", help="Show info about a saved trace")
    p.add_argument("trace_id")
    add_common_opts(p)
    p.set_defaults(func=_cmd_info)

    # assess
    p = sub.add_parser(
        "assess",
        help="Assessment: set or show a manual task & outcome assessment for a trace",
    )
    p.add_argument("trace_id")
    p.add_argument(
        "--task",
        default=None,
        help="What the agent was asked to accomplish (required when setting)",
    )
    p.add_argument(
        "--outcome",
        choices=["success", "partial", "fail", "unknown"],
        default=None,
        help="Outcome status (required when setting)",
    )
    p.add_argument(
        "--source",
        choices=["human", "test", "evaluator", "unknown"],
        default=None,
        help="Who or what determined the outcome (default: unknown)",
    )
    p.add_argument(
        "--notes",
        default=None,
        help="Free-form notes: what was produced, verified, or remains unknown",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Remove the assessment for this trace",
    )
    add_common_opts(p)
    p.set_defaults(func=_cmd_assess)

    # finding
    p = sub.add_parser(
        "finding",
        help="Findings: list, add, remove, or clear persisted findings for a trace",
    )
    finding_sub = p.add_subparsers(dest="finding_command", required=True)

    p_flist = finding_sub.add_parser("list", help="List findings for a trace")
    p_flist.add_argument("trace_id")
    add_common_opts(p_flist)
    p_flist.set_defaults(func=_cmd_finding)

    p_fadd = finding_sub.add_parser("add", help="Add a finding to a trace")
    p_fadd.add_argument("trace_id")
    p_fadd.add_argument("--title", required=True, help="Concise description of the finding")
    p_fadd.add_argument(
        "--pattern",
        required=True,
        help="Primary pattern label (see mechanisms.md; free string, vocabulary is living)",
    )
    p_fadd.add_argument(
        "--secondary-pattern",
        action="append",
        default=None,
        help="Additional pattern label (repeatable)",
    )
    p_fadd.add_argument(
        "--impact",
        choices=[
            "incorrect_result",
            "unverified_completion",
            "external_side_effect",
            "resource_exhaustion",
            "no_impact",
        ],
        default="no_impact",
        help="Consequence of the pattern (default: no_impact)",
    )
    p_fadd.add_argument(
        "--strength",
        choices=["observed", "suspected", "confirmed"],
        default="observed",
        help="Evidence strength (default: observed)",
    )
    p_fadd.add_argument(
        "--axis",
        choices=["quality", "cost", "speed"],
        default="quality",
        help="Improvement-triangle axis most affected (default: quality)",
    )
    p_fadd.add_argument(
        "--locus",
        default=None,
        help="Fault locus (free string; e.g. model, agent_harness, context, tool_integration)",
    )
    p_fadd.add_argument(
        "--evidence",
        action="append",
        default=None,
        help='Evidence reference, e.g. "<run_id>: what it shows" (repeatable)',
    )
    p_fadd.add_argument(
        "--assessment",
        default=None,
        help="What the evidence establishes, and what it does NOT establish",
    )
    p_fadd.add_argument("--next", default=None, help="Suggested next action (not a fix)")
    p_fadd.add_argument(
        "--candidate",
        default=None,
        help="Candidate improvement (labeled as candidate, not a recommendation)",
    )
    p_fadd.add_argument(
        "--validation",
        default=None,
        help="How to test whether the improvement helps",
    )
    add_common_opts(p_fadd)
    p_fadd.set_defaults(func=_cmd_finding)

    p_fremove = finding_sub.add_parser("remove", help="Remove a finding by ID")
    p_fremove.add_argument("trace_id")
    p_fremove.add_argument("finding_id", help="Finding ID (e.g. f1)")
    add_common_opts(p_fremove)
    p_fremove.set_defaults(func=_cmd_finding)

    p_fclear = finding_sub.add_parser("clear", help="Remove all findings for a trace")
    p_fclear.add_argument("trace_id")
    add_common_opts(p_fclear)
    p_fclear.set_defaults(func=_cmd_finding)
