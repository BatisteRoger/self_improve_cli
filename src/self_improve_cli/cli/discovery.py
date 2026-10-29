"""Discovery and ingestion commands — list, list-runs, list-projects, fetch.

These are the only commands that talk to LangSmith (besides `prompt pull`).
Everything else operates on locally stored sanitized traces.
"""

from __future__ import annotations

import argparse
import json
import os
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
from self_improve_cli.domain import OutcomeSource, Trace, autofill_assessment
from self_improve_cli.privacy import (
    AnonymizerBackend,
    ValueMapSpec,
    anonymize_trace,
    is_presidio_available,
    parse_drop_keys,
    parse_value_maps,
)
from self_improve_cli.representations import write_ter


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


def _privacy_env(name: str) -> str:
    """Read a privacy setting: SELF_IMPROVE_<name>, else the legacy SELFIIMPROVE_<name>.

    The misspelled SELFIIMPROVE_ prefix stays supported so existing .env files
    keep working; the correctly spelled prefix wins when both are set.
    """
    for prefix in ("SELF_IMPROVE_", "SELFIIMPROVE_"):
        val = os.environ.get(prefix + name, "").strip()
        if val:
            return val
    return ""


def _resolve_anonymizer(args: argparse.Namespace) -> AnonymizerBackend:
    """Resolve the anonymizer backend: CLI flag overrides env var, env var overrides 'auto'.

    Valid values: "auto", "presidio", "regex".
    """
    cli_val = getattr(args, "anonymizer", None)
    if cli_val:
        return cli_val  # type: ignore[return-value]
    env_val = _privacy_env("ANONYMIZER").lower()
    if env_val in ("auto", "presidio", "regex"):
        return env_val  # type: ignore[return-value]
    return "auto"


def _resolve_structural_policy() -> tuple[frozenset[str], tuple[ValueMapSpec, ...]]:
    """Read DROP_KEYS and VALUE_MAP from the environment. Raises ValueError on a bad spec."""
    return parse_drop_keys(_privacy_env("DROP_KEYS")), parse_value_maps(_privacy_env("VALUE_MAP"))


def _autofill_assessment(source: Any, store: Any, trace: Trace) -> str:
    """Pre-fill assessment.json from the sanitized trace and recorded feedback.

    Never touches a human assessment; otherwise fills empty fields only.
    Returns a short status for the fetch output.
    """
    from datetime import UTC, datetime

    existing = store.load_assessment(trace.trace_id)
    if existing is not None and existing.outcome_source == OutcomeSource.HUMAN:
        return "kept (source=human)"
    try:
        feedback = source.fetch_feedback(trace.trace_id)
    except Exception as exc:  # noqa: BLE001 — feedback is optional; the trace is already saved
        print(
            f"Warning: feedback lookup failed ({type(exc).__name__}); outcome left unknown.",
            file=sys.stderr,
        )
        feedback = None
    updated = autofill_assessment(existing, trace, feedback)
    if updated is None:
        return "unchanged"
    updated.assessed_at = datetime.now(UTC).isoformat()
    store.save_assessment(updated)
    return "created" if existing is None else "updated"


def _cmd_fetch_experiment(args: argparse.Namespace) -> int:
    """Fetch every run of an experiment session and write experiment.json."""
    from datetime import UTC, datetime

    from self_improve_cli.domain import outcome_from_policy, parse_outcome_key
    from self_improve_cli.sources.langsmith import LangSmithSource

    policy = parse_outcome_key(args.outcome_key) if args.outcome_key else None
    backend = _resolve_anonymizer(args)
    drop_keys, value_maps = _resolve_structural_policy()
    source = LangSmithSource(project_name=args.project)
    listing = source.fetch_experiment(args.experiment)
    store = _get_store(args)
    examples: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, str]] = []
    counts = {"stored": 0, "skipped": 0, "failed": 0}
    outcomes = {"success": 0, "fail": 0, "unknown": 0}
    for ref in listing.runs:
        group = examples.setdefault(ref.example_id, [])
        if (store.traces_dir / ref.trace_id / "sanitized.json").exists():
            counts["skipped"] += 1  # sanitized traces are immutable: resume skips them
            trace = store.load_trace(ref.trace_id)
        else:
            try:
                trace = source.fetch_trace(ref.trace_id, project_id=listing.session_id)
                if not trace.runs:
                    raise ValueError("no runs returned")
            except Exception as exc:  # noqa: BLE001 — one bad run must not abort the set
                counts["failed"] += 1
                reason = f"{type(exc).__name__}: {exc}"[:200]
                failures.append(
                    {
                        "run_id": ref.run_id,
                        "example_id": ref.example_id,
                        "stage": "trace_fetch",
                        "reason": reason,
                    }
                )
                continue
            if args.keep_raw:
                store.save_raw(trace)
            trace, _ = anonymize_trace(
                trace, backend=backend, drop_keys=drop_keys, value_maps=value_maps
            )
            store.save_sanitized(trace)
            write_ter(ref.trace_id, trace.runs, store)
            counts["stored"] += 1
        notes = {
            f.key: f"score is the mean of {f.n} feedback points, thresholded under "
            f"{policy.describe()}"
            for f in ref.feedback
            if policy and policy.threshold is not None and f.key == policy.key and (f.n or 0) > 1
        }
        store.save_feedback(ref.trace_id, ref.feedback, notes)
        existing = store.load_assessment(ref.trace_id)
        updated = autofill_assessment(existing, trace, ref.feedback, policy=policy)
        if updated is not None:
            updated.assessed_at = datetime.now(UTC).isoformat()
            store.save_assessment(updated)
        outcome = outcome_from_policy(ref.feedback, policy)[0].value
        outcomes[outcome] += 1
        group.append(
            {
                "run_id": ref.run_id,
                "trace_id": ref.trace_id,
                "status": ref.status,
                "outcome": outcome,
            }
        )
    complete = listing.complete and not failures
    receipt = {
        "schema_version": 1,
        "source": "langsmith",
        "experiment_name": listing.name,
        "session_id": listing.session_id,
        "dataset_id": listing.dataset_id,
        "fetched_at": datetime.now(UTC).isoformat(),
        "complete": complete,
        "outcome_policy": policy.to_dict() if policy else None,
        "conditions": listing.conditions,
        "examples": examples,
        "failures": failures,
    }
    path = store.save_experiment(receipt)
    sizes = [len(v) for v in examples.values()] or [0]
    _output(
        {
            "experiment_name": listing.name,
            "session_id": listing.session_id,
            "dataset_id": listing.dataset_id,
            "complete": complete,
            "examples": len(examples),
            "runs_per_example": {"min": min(sizes), "max": max(sizes)},
            "traces": counts,
            "outcomes": outcomes,
            "outcome_policy": receipt["outcome_policy"],
            "anonymizer_backend": backend,
            "receipt_path": str(path),
            "next": [f"self-improve gate --baseline '{listing.name}' --candidate <other>"],
        },
        args,
    )
    if not complete:
        print(
            "Warning: experiment fetch incomplete; re-run the same command to resume.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    return EXIT_OK


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch a trace, anonymize it, and persist the sanitized version."""
    from self_improve_cli.sources.langsmith import LangSmithSource

    usage = None
    if bool(args.experiment) == bool(args.trace_id):
        usage = (
            "pass either `fetch <trace_id>` or `fetch --experiment <name>`, not both or neither."
        )
    elif args.experiment and args.from_run:
        usage = "--from-run cannot be combined with --experiment."
    elif args.outcome_key and not args.experiment:
        usage = "--outcome-key is only valid with --experiment."
    if usage:
        print(f"Error: {usage}", file=sys.stderr)
        return EXIT_USAGE
    if args.experiment:
        return _cmd_fetch_experiment(args)

    # Resolve privacy settings before any network call: a bad spec fails closed.
    backend = _resolve_anonymizer(args)
    drop_keys, value_maps = _resolve_structural_policy()

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
    warning: str | None = None
    if not trace.runs:
        if project_id:
            warning = (
                f"No runs found for trace {trace_id} in project_id={project_id}. "
                "The trace may be empty or access may be restricted."
            )
        else:
            where = f" in project '{source.project_name}'" if source.project_name else ""
            warning = (
                f"No runs found for trace {trace_id}{where}. The trace may live "
                "in a different project than the configured one. Try `self-improve "
                "list-projects` to discover project names, then `self-improve "
                "fetch <trace_id> --project <name>`."
            )
        print(f"Warning: {warning}", file=sys.stderr)

    # Save raw BEFORE anonymization (anonymize_trace mutates in place).
    store = _get_store(args)
    if args.keep_raw:
        store.save_raw(trace)

    # Anonymize before persistence (privacy boundary).
    trace, report = anonymize_trace(
        trace, backend=backend, drop_keys=drop_keys, value_maps=value_maps
    )

    sanitized_path = store.save_sanitized(trace)

    stats = write_ter(trace_id, trace.runs, store)

    # Task comes from the sanitized trace — never from raw data.
    assessment_status = _autofill_assessment(source, store, trace)

    result: dict[str, Any] = {
        "trace_id": trace_id,
        "runs": len(trace.runs),
        "sanitized": True,
        "sanitization_report": report.to_dict(),
        "anonymizer_backend": backend,
        "presidio_available": is_presidio_available(),
        "sanitized_path": str(sanitized_path),
        "raw_saved": args.keep_raw,
        "ter_stats": stats,
        "assessment": assessment_status,
    }
    if warning:
        result["warning"] = warning
    if getattr(args, "from_run", False):
        result["resolved_from_run"] = args.trace_id
    _output(result, args)
    return EXIT_OK


def register(sub: argparse._SubParsersAction) -> None:
    """Register discovery commands on the given subparsers action."""
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
        "parent trace ID before fetching. Tries the configured project first "
        "(SmithDB); falls back to a legacy global lookup if the run is not in "
        "the configured project. Use --project to target a different project.",
    )
    p.add_argument(
        "--anonymizer",
        choices=["auto", "presidio", "regex"],
        default=None,
        help="Anonymization backend: 'auto' (presidio if installed, else regex), "
        "'presidio' (force, errors if unavailable), 'regex' (skip presidio, faster). "
        "Overrides SELF_IMPROVE_ANONYMIZER env var. Default: auto.",
    )
    p.add_argument(
        "--experiment",
        default=None,
        help="Fetch every run of a LangSmith experiment session (by name); writes "
        "data/experiments/<session_id>/experiment.json grouping trace IDs by example.",
    )
    p.add_argument(
        "--outcome-key",
        default=None,
        help="With --experiment: feedback key deciding pass/fail. 'KEY' needs a binary "
        "0/1 score; 'KEY>=T' thresholds a continuous score. Missing key -> unknown.",
    )
    p.add_argument("trace_id", nargs="?", default=None)
    add_common_opts(p)
    p.set_defaults(func=_cmd_fetch)
