"""Experiment-set gate — a per-example flip verdict above a measured noise floor.

Compares a baseline set and a candidate set of runs over the same dataset
examples, each example repeated k times per side.

- **Classification** per example and side: `stable_pass` (all pass),
  `stable_fail` (all fail), `unstable` (mixed). Any `unknown` run makes the
  example `unknown`; k = 1 makes it `single` (no noise estimate, never a
  pairwise fallback).
- **Flips**: regression = `stable_pass -> stable_fail`, gain = the reverse.
  The full 3x3 transition table is reported.
- **Noise floor** = share of `unstable` examples in the baseline (examples
  with k >= 2 and no unknown run). APPROXIMATE: with small k an unstable
  example can show k identical outcomes by chance, so the floor
  underestimates true instability. Not a significance test.
- **Quality rule**, first match: two outcome rules (key vs key, or feedback
  vs `assess`) or an incomplete experiment fetch -> `quality_unknown`; no
  comparable example -> inconclusive; regressions / comparable > floor ->
  fail; an example with no run on one side -> `insufficient_coverage`;
  stable -> unstable moves / comparable > floor -> `destabilized`; no
  regression -> pass; else `within_noise`. Gains never offset regressions.
- **Efficiency** (APPROXIMATE): per comparable example, the candidate median
  is compared with the baseline min-max range; a metric is degraded when
  more examples went up than down. Missing values are unavailable, never 0;
  a metric measured nowhere is `unavailable`, not equal.
- **Verdict** is conjunctive per criterion: any fail -> fail; else any
  inconclusive -> inconclusive with the highest-priority reason code; else
  pass.

Pure functions over prepared `Side` objects: no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from self_improve_cli.domain.experiment import OutcomePolicy
from self_improve_cli.metrics.efficiency import EfficiencySummary
from self_improve_cli.metrics.gate import ReasonCode, Verdict

CLASSES = ("stable_pass", "unstable", "stable_fail")
REASON_PRIORITY = [
    ReasonCode.QUALITY_UNKNOWN,
    ReasonCode.INSUFFICIENT_COVERAGE,
    ReasonCode.NO_NOISE_ESTIMATE,
    ReasonCode.DESTABILIZED,
    ReasonCode.WITHIN_NOISE,
    ReasonCode.CONDITIONS_DIFFER,
    ReasonCode.EFFICIENCY_TRADEOFF,
]
METRICS = ("tokens", "latency_ms", "tool_calls")
MAX_REPRESENTATIVE = 3
MAX_MARKDOWN_ROWS = 10

SETS_RULE = (
    "per example: stable_pass / stable_fail / unstable over k runs (unknown run -> unknown, "
    "k=1 -> no_noise_estimate); noise floor = baseline unstable share; "
    "policy mismatch or incomplete fetch -> quality_unknown; "
    "regressions/comparable > floor -> fail; example missing on a side -> "
    "insufficient_coverage; destabilized/comparable > floor -> destabilized; "
    "0 regressions -> pass; else within_noise; human assessments override feedback; "
    "conjunctive over quality, efficiency (and conditions with --strict-conditions)"
)


@dataclass
class RunOutcome:
    """One run of a side: its trace, binary outcome, and efficiency totals."""

    trace_id: str
    outcome: str  # pass | fail | unknown
    efficiency: EfficiencySummary | None = None


@dataclass
class Side:
    """A set of runs grouped by dataset example (an experiment or an id file)."""

    label: str
    kind: str  # experiment | file
    ref: str
    policy: OutcomePolicy | None
    groups: dict[str, list[RunOutcome]]
    conditions: dict[str, Any] = field(default_factory=dict)
    dataset_id: str | None = None
    complete: bool | None = None  # None: not recorded (id files, older receipts)
    failures: int | None = None
    # which evidence decided pass/fail: basis, human_overrides, description
    outcomes: dict[str, Any] = field(default_factory=dict)


def classify(runs: list[RunOutcome] | None) -> str:
    """Classify an example's runs on one side."""
    if not runs:
        return "missing"
    outcomes = {r.outcome for r in runs}
    if "unknown" in outcomes:
        return "unknown"
    if len(runs) == 1:
        return "single"
    if outcomes == {"pass"}:
        return "stable_pass"
    if outcomes == {"fail"}:
        return "stable_fail"
    return "unstable"


def _transition(b: str, c: str) -> str:
    if (b, c) == ("stable_pass", "stable_fail"):
        return "regression"
    if (b, c) == ("stable_fail", "stable_pass"):
        return "gain"
    return "same" if b == c else "move"


def example_rows(baseline: Side, candidate: Side) -> list[dict[str, Any]]:
    """One row per example in the union of both sides."""
    rows: list[dict[str, Any]] = []
    for ex in sorted(set(baseline.groups) | set(candidate.groups)):
        b, c = classify(baseline.groups.get(ex)), classify(candidate.groups.get(ex))
        row: dict[str, Any] = {"example_id": ex, "baseline": b, "candidate": c}
        if "missing" in (b, c):
            row["reason_code"] = ReasonCode.INSUFFICIENT_COVERAGE.value
        elif "unknown" in (b, c):
            row["reason_code"] = ReasonCode.QUALITY_UNKNOWN.value
        elif "single" in (b, c):
            row["reason_code"] = ReasonCode.NO_NOISE_ESTIMATE.value
        else:
            row["transition"] = _transition(b, c)
        rows.append(row)
    return rows


def noise_floor(side: Side) -> tuple[float | None, int, int]:
    """(floor, unstable count, classified count) over a side's examples.

    Takes any `Side` so a dedicated A/A anchor set can supply the floor later.
    """
    classes = [classify(runs) for runs in side.groups.values()]
    classified = [c for c in classes if c in CLASSES]
    unstable = classified.count("unstable")
    if not classified:
        return None, 0, 0
    return unstable / len(classified), unstable, len(classified)


def _policy_mismatch(baseline: Side, candidate: Side) -> str | None:
    """Two outcome rules: different keys/thresholds, or feedback vs `assess` outcomes."""
    pb, pc = baseline.policy, candidate.policy
    if pb is None or pc is None:
        return None
    mixed = (pb.key is None) != (pc.key is None)
    if mixed or (pb.key is not None and not pb.same_rule(pc)):
        return f"baseline {pb.describe()} vs candidate {pc.describe()}"
    return None


def quality_criterion(
    baseline: Side, candidate: Side, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the quality rule (module docstring) to the example rows."""
    table = {b: dict.fromkeys(CLASSES, 0) for b in CLASSES}
    comparable = [r for r in rows if "transition" in r]
    for r in comparable:
        table[r["baseline"]][r["candidate"]] += 1
    regressions = sum(1 for r in comparable if r["transition"] == "regression")
    gains = sum(1 for r in comparable if r["transition"] == "gain")
    destabilized = sum(
        1 for r in comparable if r["candidate"] == "unstable" and r["baseline"] != "unstable"
    )
    missing = sum(1 for r in rows if "missing" in (r["baseline"], r["candidate"]))
    floor, unstable, classified = noise_floor(baseline)
    n = len(comparable)
    result: dict[str, Any] = {
        "noise_floor": None if floor is None else round(floor, 4),
        "noise_floor_basis": {"unstable": unstable, "classified": classified},
        "comparable": n,
        "regressions": regressions,
        "gains": gains,
        "destabilized": destabilized,
        "missing": missing,
        "regression_rate": round(regressions / n, 4) if n else None,
        "gain_rate": round(gains / n, 4) if n else None,
        "destabilized_rate": round(destabilized / n, 4) if n else None,
        "transitions": table,
        "inconclusive_examples": [
            {"example_id": r["example_id"], "reason_code": r["reason_code"]}
            for r in rows
            if "reason_code" in r
        ],
        "approximate": True,
    }
    mismatch = _policy_mismatch(baseline, candidate)
    incomplete = [s.label for s in (baseline, candidate) if s.complete is False]
    if mismatch:
        status, code = Verdict.INCONCLUSIVE, ReasonCode.QUALITY_UNKNOWN
        reason = (
            f"Outcome policies differ ({mismatch}); flips are not comparable. "
            "Pass --outcome-key to gate to re-derive both sides under one policy."
        )
    elif incomplete:
        status, code = Verdict.INCONCLUSIVE, ReasonCode.QUALITY_UNKNOWN
        reason = (
            f"Incomplete experiment fetch ({', '.join(incomplete)}): missing runs can fake "
            "stability or flips. Re-run `self-improve fetch --experiment` to resume."
        )
    elif n == 0:
        status = Verdict.INCONCLUSIVE
        row_codes = {r["reason_code"] for r in rows if "reason_code" in r}
        code = next(
            (c for c in REASON_PRIORITY if c.value in row_codes), ReasonCode.QUALITY_UNKNOWN
        )
        reason = "No example is comparable on both sides (see inconclusive_examples)."
    else:
        # a comparable row implies a classified baseline example, so the floor exists
        assert floor is not None
        if regressions / n > floor:
            status, code = Verdict.FAIL, None
            reason = (
                f"{regressions}/{n} regressions ({regressions / n:.1%}) exceed the noise "
                f"floor ({floor:.1%})."
            )
        elif missing:
            status, code = Verdict.INCONCLUSIVE, ReasonCode.INSUFFICIENT_COVERAGE
            reason = (
                f"{missing} example(s) have no run on one side; quality cannot pass on "
                "partial coverage (see inconclusive_examples)."
            )
        elif destabilized / n > floor:
            status, code = Verdict.INCONCLUSIVE, ReasonCode.DESTABILIZED
            reason = (
                f"{destabilized}/{n} stable examples became unstable ({destabilized / n:.1%}), "
                f"above the noise floor ({floor:.1%})."
            )
        elif regressions == 0:
            status, code = Verdict.PASS, None
            reason = f"No stable regression over {n} comparable examples."
        else:
            status, code = Verdict.INCONCLUSIVE, ReasonCode.WITHIN_NOISE
            reason = (
                f"{regressions}/{n} regressions ({regressions / n:.1%}) do not exceed the "
                f"noise floor ({floor:.1%})."
            )
        reason += f" Destabilized (stable -> unstable): {destabilized}/{n}."
    improved = status == Verdict.PASS and floor is not None and gains / n > floor
    result.update(
        status=status.value,
        reason_code=code.value if code else None,
        reason=reason,
        improved=improved,
    )
    return result


def _metric_value(eff: EfficiencySummary | None, name: str) -> int | None:
    if eff is None:
        return None
    if name == "tokens":
        return eff.tokens
    if name == "latency_ms":
        return eff.latency_ms
    return eff.total_tool_calls


def _values(runs: list[RunOutcome] | None, name: str) -> list[int] | None:
    vals = [_metric_value(r.efficiency, name) for r in runs or []]
    if not vals or any(v is None for v in vals):
        return None
    return [v for v in vals if v is not None]


def efficiency_criterion(
    baseline: Side, candidate: Side, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-metric counts of comparable examples whose candidate median left the baseline range.

    A metric measured on no example is `unavailable`, never `equal`; when every
    metric is unavailable the criterion is `unavailable` and does not count as a pass.
    """
    examples = [r["example_id"] for r in rows if "transition" in r]
    metrics: dict[str, dict[str, Any]] = {}
    for name in METRICS:
        counts = {"up": 0, "down": 0, "within_spread": 0, "unavailable": 0}
        deltas: list[float] = []
        for ex in examples:
            b = _values(baseline.groups.get(ex), name)
            c = _values(candidate.groups.get(ex), name)
            if b is None or c is None or len(b) < 2:
                counts["unavailable"] += 1
                continue
            cm = median(c)
            deltas.append(cm - median(b))
            if cm > max(b):
                counts["up"] += 1
            elif cm < min(b):
                counts["down"] += 1
            else:
                counts["within_spread"] += 1
        if not deltas:
            direction = "unavailable"
        elif counts["up"] > counts["down"]:
            direction = "degraded"
        elif counts["down"] > counts["up"]:
            direction = "improved"
        else:
            direction = "equal"
        metrics[name] = {
            **counts,
            "direction": direction,
            "median_delta": median(deltas) if deltas else None,
        }
    degraded = [n for n, m in metrics.items() if m["direction"] == "degraded"]
    unavailable = [n for n, m in metrics.items() if m["direction"] == "unavailable"]
    if degraded:
        status, code = Verdict.INCONCLUSIVE.value, ReasonCode.EFFICIENCY_TRADEOFF
        reason = f"Degraded beyond the baseline spread: {', '.join(degraded)}."
    elif len(unavailable) == len(METRICS):
        status, code = "unavailable", None
        reason = "No metric measured on a comparable example with baseline k >= 2."
    else:
        status, code = Verdict.PASS.value, None
        reason = "No metric degraded beyond the baseline spread."
        if unavailable:
            reason += f" Unavailable: {', '.join(unavailable)}."
    return {
        "status": status,
        "reason_code": code.value if code else None,
        "reason": reason,
        "approximate": True,
        "metrics": metrics,
    }


def _state(b: Any, c: Any) -> str:
    if b in (None, [], "") or c in (None, [], ""):
        return "unavailable"
    return "same" if b == c else "differs"


def _policy_str(p: OutcomePolicy | None) -> str | None:
    return p.describe() if p else None


def _gap(baseline: Side, candidate: Side) -> str | None:
    from datetime import datetime

    b = baseline.conditions.get("end_time") or baseline.conditions.get("start_time")
    c = candidate.conditions.get("start_time")
    if not b or not c:
        return None
    try:
        delta = datetime.fromisoformat(str(c).replace("Z", "+00:00")) - datetime.fromisoformat(
            str(b).replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        return None
    return f"{delta.total_seconds() / 3600:+.1f}h"


def condition_facts(baseline: Side, candidate: Side) -> list[dict[str, Any]]:
    """Confounds as facts: same / differs / unavailable, never judged."""
    bc, cc = baseline.conditions, candidate.conditions
    facts = []
    for name, b, c in [
        ("models", bc.get("models"), cc.get("models")),
        (
            "git_commit",
            bc.get("git_commit") or bc.get("revision_id"),
            cc.get("git_commit") or cc.get("revision_id"),
        ),
        ("git_dirty", bc.get("git_dirty"), cc.get("git_dirty")),
        ("dataset_version", bc.get("dataset_version"), cc.get("dataset_version")),
        ("outcome_policy", _policy_str(baseline.policy), _policy_str(candidate.policy)),
    ]:
        facts.append({"name": name, "baseline": b, "candidate": c, "state": _state(b, c)})
    gap = _gap(baseline, candidate)
    facts.append(
        {
            "name": "wall_clock_gap",
            "baseline": bc.get("start_time"),
            "candidate": cc.get("start_time"),
            "state": "info" if gap else "unavailable",
            "value": gap,
            "inferred": baseline.kind == "file" or candidate.kind == "file",
        }
    )
    facts.append(
        {
            "name": "dataset_version_tag",
            "baseline": None,
            "candidate": None,
            "state": "unavailable",
            "value": "not recorded by the source",
        }
    )
    return facts


def _representative(rows: list[dict[str, Any]], baseline: Side, candidate: Side) -> list[dict]:
    out = []
    for r in [r for r in rows if r.get("transition") == "regression"][:MAX_REPRESENTATIVE]:
        b = sorted(x.trace_id for x in baseline.groups[r["example_id"]])[0]
        c = sorted(x.trace_id for x in candidate.groups[r["example_id"]])[0]
        out.append(
            {
                "example_id": r["example_id"],
                "baseline_trace": b,
                "candidate_trace": c,
                "next": [
                    f"self-improve diff {b} {c}",
                    f"self-improve narrative {c} --budget 4000",
                ],
            }
        )
    return out


def _side_summary(side: Side) -> dict[str, Any]:
    sizes = [len(v) for v in side.groups.values()] or [0]
    return {
        "label": side.label,
        "kind": side.kind,
        "ref": side.ref,
        "examples": len(side.groups),
        "runs": sum(sizes),
        "k": {"min": min(sizes), "max": max(sizes)},
        "outcome_policy": side.policy.to_dict() if side.policy else None,
        "complete": side.complete,
        "failures": side.failures,
        "outcomes": side.outcomes or None,
    }


def _completeness(side: dict[str, Any]) -> str:
    if side["complete"] is None:
        return "completeness unavailable"
    failures = "unavailable" if side["failures"] is None else side["failures"]
    return f"{'complete' if side['complete'] else 'INCOMPLETE'}, failures {failures}"


def _outcomes_line(side: dict[str, Any]) -> str:
    return (side["outcomes"] or {}).get("description") or "source unavailable"


def gate_sets_data(
    baseline: Side, candidate: Side, strict_conditions: bool = False
) -> dict[str, Any]:
    """Build the structured experiment-set gate receipt."""
    rows = example_rows(baseline, candidate)
    quality = quality_criterion(baseline, candidate, rows)
    efficiency = efficiency_criterion(baseline, candidate, rows)
    facts = condition_facts(baseline, candidate)
    differing = [f["name"] for f in facts if f["state"] == "differs"]
    conditions: dict[str, Any] = {"status": "info", "reason_code": None, "facts": facts}
    if strict_conditions:
        if differing:
            conditions.update(
                status=Verdict.INCONCLUSIVE.value,
                reason_code=ReasonCode.CONDITIONS_DIFFER.value,
                reason=f"--strict-conditions: {', '.join(differing)} differ.",
            )
        else:
            conditions.update(status=Verdict.PASS.value, reason="No condition differs.")
    criteria = {"quality": quality, "efficiency": efficiency, "conditions": conditions}
    statuses = [c["status"] for c in criteria.values()]
    codes = {c["reason_code"] for c in criteria.values() if c["reason_code"]}
    if Verdict.FAIL.value in statuses:
        verdict, code, reason = Verdict.FAIL, None, quality["reason"]
    elif Verdict.INCONCLUSIVE.value in statuses:
        verdict = Verdict.INCONCLUSIVE
        code = next(c for c in REASON_PRIORITY if c.value in codes)
        reason = next(c["reason"] for c in criteria.values() if c["reason_code"] == code.value)
    else:
        verdict, code, reason = Verdict.PASS, None, quality["reason"]
    return {
        "mode": "sets",
        "baseline": _side_summary(baseline),
        "candidate": _side_summary(candidate),
        "verdict": verdict.value,
        "reason_code": code.value if code else None,
        "reason": reason,
        "rule": SETS_RULE,
        "criteria": criteria,
        "examples": rows,
        "representative_regressions": _representative(rows, baseline, candidate),
    }


def build_gate_sets(data: dict[str, Any]) -> str:
    """Render the set gate receipt as compact Markdown."""
    q = data["criteria"]["quality"]
    e = data["criteria"]["efficiency"]
    cond = data["criteria"]["conditions"]
    b, c = data["baseline"], data["candidate"]
    head = f"# Gate (sets): {data['verdict'].upper()}"
    if data["reason_code"]:
        head += f" ({data['reason_code']})"
    floor = "unknown" if q["noise_floor"] is None else f"{q['noise_floor']:.1%}"
    basis = q["noise_floor_basis"]
    lines = [
        head,
        "",
        f"Baseline: {b['label']} ({b['examples']} examples, k={b['k']['min']}-{b['k']['max']}, "
        f"{_completeness(b)})",
        f"Candidate: {c['label']} ({c['examples']} examples, k={c['k']['min']}-{c['k']['max']}, "
        f"{_completeness(c)})",
        f"Baseline outcomes: {_outcomes_line(b)}",
        f"Candidate outcomes: {_outcomes_line(c)}",
        f"Verdict reason: {data['reason']}",
        "",
        f"- Quality (APPROXIMATE): {q['status']}"
        + (f" ({q['reason_code']})" if q["reason_code"] else ""),
        f"  - noise floor {floor} ({basis['unstable']}/{basis['classified']} baseline unstable)"
        f" | regressions {q['regressions']}/{q['comparable']} | gains {q['gains']}/"
        f"{q['comparable']} | destabilized {q['destabilized']}/{q['comparable']}"
        f" | missing on one side {q['missing']}",
        f"  - {q['reason']}",
        "",
        "| baseline \\ candidate | " + " | ".join(CLASSES) + " |",
        "| --- | ---: | ---: | ---: |",
        *(
            f"| {row} | " + " | ".join(str(q["transitions"][row][col]) for col in CLASSES) + " |"
            for row in CLASSES
        ),
        "",
        f"- Efficiency (APPROXIMATE): {e['status']}",
        *(
            f"  - {name}: {m['direction']} (up {m['up']}, down {m['down']}, within "
            f"{m['within_spread']}, unavailable {m['unavailable']})"
            for name, m in e["metrics"].items()
        ),
        f"- Conditions: {cond['status']}",
        *(
            f"  - {f['name']}: {f['state']}"
            + (f" ({f['baseline']} vs {f['candidate']})" if f["state"] == "differs" else "")
            + (f" {f['value']}" if f.get("value") and f["state"] == "info" else "")
            for f in cond["facts"]
        ),
    ]
    flagged = [
        r for r in data["examples"] if not (r["baseline"] == r["candidate"] == "stable_pass")
    ]
    if flagged:
        lines += ["", "Examples (non stable_pass -> stable_pass):"]
        for r in flagged[:MAX_MARKDOWN_ROWS]:
            tag = r.get("transition") or r.get("reason_code")
            lines.append(f"- {r['example_id']}: {r['baseline']} -> {r['candidate']} [{tag}]")
        if len(flagged) > MAX_MARKDOWN_ROWS:
            lines.append(f"- [omitted {len(flagged) - MAX_MARKDOWN_ROWS} rows; use --format json]")
    if data["representative_regressions"]:
        lines += ["", "Representative regressions:"]
        for r in data["representative_regressions"]:
            lines.append(f"- {r['example_id']}: {r['baseline_trace']} -> {r['candidate_trace']}")
            lines += [f"  - `{cmd}`" for cmd in r["next"]]
    lines += ["", f"Rule: {data['rule']}"]
    return "\n".join(lines)
