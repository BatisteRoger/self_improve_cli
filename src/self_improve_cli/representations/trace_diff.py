"""Trajectory diff — align two traces step by step and show where they diverge.

A slicing view over existing data: no scoring, no new metrics. Steps are the
narrative steps (significant runs after the root, 1-based), so step N here is
step N in `narrative`. Alignment runs `difflib.SequenceMatcher` over a per-step
signature `(run_type, name)`: deterministic, stdlib, and its opcodes map
directly onto aligned pairs, divergence blocks, and only-in-A/B steps.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from self_improve_cli.domain import Run, RunType
from self_improve_cli.representations.common import _latency_s, _truncate, significant_runs

_MAX_DIFF_CHARS = 200
# Steps listed per side of a divergence block before an omission marker.
_MAX_BLOCK_STEPS = 10


def _steps(runs: list[Run], trace_id: str) -> tuple[Run, list[Run]]:
    """Root run and narrative steps (1-based: steps[0] is step 1)."""
    sig = significant_runs(runs)
    if len(sig) < 2:
        raise ValueError(
            f"Trace '{trace_id}' has no steps to align. "
            f"Inspect it with `self-improve skeleton {trace_id}`."
        )
    return sig[0], sig[1:]


def _signature(run: Run) -> tuple[str, str]:
    return (run.run_type.value, run.name)


def _decision(run: Run) -> str:
    """What the step did: tool inputs, or the tool calls an LLM step emitted."""
    if run.run_type == RunType.LLM:
        out = run.output_message
        calls = [{"name": tc.name, "args": tc.args} for tc in out.tool_calls] if out else []
        return json.dumps(calls, default=str, ensure_ascii=False, sort_keys=True)
    if run.run_type == RunType.TOOL:
        return json.dumps(run.inputs, default=str, ensure_ascii=False, sort_keys=True)
    return ""


def _delta(a: float | int | None, b: float | int | None) -> float | int | None:
    if a is None or b is None:
        return None
    return round(b - a, 3) if isinstance(a, float) or isinstance(b, float) else b - a


def _pair_changes(a: Run, b: Run) -> dict[str, Any]:
    """Recorded differences between two aligned steps (empty dict = unchanged)."""
    changes: dict[str, Any] = {}
    if (a.status or "") != (b.status or ""):
        changes["status"] = {"a": a.status, "b": b.status}
    if (a.error or "") != (b.error or ""):
        changes["error"] = {
            "a": _truncate(str(a.error), _MAX_DIFF_CHARS) if a.error else None,
            "b": _truncate(str(b.error), _MAX_DIFF_CHARS) if b.error else None,
        }
    dec_a, dec_b = _decision(a), _decision(b)
    if dec_a != dec_b:
        key = "tool_calls" if a.run_type == RunType.LLM else "args"
        changes[key] = {
            "a": _truncate(dec_a, _MAX_DIFF_CHARS),
            "b": _truncate(dec_b, _MAX_DIFF_CHARS),
        }
    return changes


def _step_ref(step: int, run: Run) -> dict[str, Any]:
    return {
        "step": step,
        "run_id": run.id,
        "run_type": run.run_type.value,
        "name": run.name,
        "status": run.status,
        "error": _truncate(str(run.error), _MAX_DIFF_CHARS) if run.error else None,
    }


def trace_diff_data(
    runs_a: list[Run], runs_b: list[Run], trace_a: str, trace_b: str
) -> dict[str, Any]:
    """Structured trajectory diff for JSON output (composable contract).

    Raises ValueError (recoverable) when the IDs are the same or a trace has
    no steps. Each segment is either `aligned` (same signature on both sides,
    with recorded changes and token/latency deltas) or `divergence`
    (consecutive steps present on one side or replaced by different steps).
    """
    if trace_a == trace_b:
        raise ValueError(
            f"Both arguments are trace '{trace_a}'. Pass two different trace IDs "
            "(see `self-improve list`)."
        )
    root_a, steps_a = _steps(runs_a, trace_a)
    root_b, steps_b = _steps(runs_b, trace_b)

    matcher = SequenceMatcher(
        None, [_signature(r) for r in steps_a], [_signature(r) for r in steps_b], autojunk=False
    )
    segments: list[dict[str, Any]] = []
    only_in_a: list[dict[str, Any]] = []
    only_in_b: list[dict[str, Any]] = []
    aligned = changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2), strict=True):
                a, b = steps_a[i], steps_b[j]
                changes = _pair_changes(a, b)
                aligned += 1
                changed += bool(changes)
                segments.append(
                    {
                        "kind": "aligned",
                        "a": _step_ref(i + 1, a),
                        "b": _step_ref(j + 1, b),
                        "changes": changes,
                        "token_delta": _delta(a.total_tokens, b.total_tokens),
                        "latency_delta_s": _delta(_latency_s(a), _latency_s(b)),
                    }
                )
        else:
            side_a = [_step_ref(i + 1, steps_a[i]) for i in range(i1, i2)]
            side_b = [_step_ref(j + 1, steps_b[j]) for j in range(j1, j2)]
            only_in_a += side_a
            only_in_b += side_b
            segments.append({"kind": "divergence", "a": side_a, "b": side_b})

    notes = []
    if root_a.name != root_b.name:
        notes.append(
            f"Root runs differ (A: {root_a.name}, B: {root_b.name}). The traces may come "
            "from different agents; alignment uses step signatures only."
        )
    return {
        "trace_a": trace_a,
        "trace_b": trace_b,
        "alignment": "difflib.SequenceMatcher over (run_type, name) per narrative step",
        "summary": {
            "steps_a": len(steps_a),
            "steps_b": len(steps_b),
            "aligned": aligned,
            "aligned_changed": changed,
            "divergence_blocks": sum(1 for s in segments if s["kind"] == "divergence"),
            "only_in_a": len(only_in_a),
            "only_in_b": len(only_in_b),
            "identical": changed == 0 and not only_in_a and not only_in_b,
        },
        "notes": notes,
        "segments": segments,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
    }


def _ref_line(side: str, ref: dict[str, Any]) -> str:
    error = f" error={ref['error']}" if ref["error"] else ""
    return (
        f"- {side}#{ref['step']} [{ref['run_type']}] {ref['name']} id={ref['run_id']} "
        f"— {ref['status'] or '?'}{error}"
    )


def _block_lines(side: str, refs: list[dict[str, Any]], trace_id: str, full: bool) -> list[str]:
    if not refs:
        return [f"- {side}: (no steps)"]
    shown = refs if full else refs[:_MAX_BLOCK_STEPS]
    lines = [_ref_line(side, r) for r in shown]
    if len(refs) > len(shown):
        lo, hi = refs[len(shown)]["step"], refs[-1]["step"]
        lines.append(
            f"- [omitted {len(refs) - len(shown)} {side} step(s) — inspect with "
            f"`self-improve narrative {trace_id} --from {lo} --to {hi}` or `--full`]"
        )
    return lines


def _fmt_delta(value: float | int | None, unit: str) -> str:
    return "n/a" if value is None else f"{value:+}{unit}"


def build_trace_diff(
    runs_a: list[Run], runs_b: list[Run], trace_a: str, trace_b: str, *, full: bool = False
) -> str:
    """Markdown trajectory diff.

    Compact by default: runs of unchanged aligned steps collapse to one line,
    and divergence blocks list at most `_MAX_BLOCK_STEPS` steps per side with
    an omission marker. `full` lists every aligned pair and every step.
    """
    data = trace_diff_data(runs_a, runs_b, trace_a, trace_b)
    s = data["summary"]
    lines = [
        f"# Trajectory diff — A={trace_a} vs B={trace_b}",
        "",
        f"Steps A={s['steps_a']} B={s['steps_b']}: {s['aligned']} aligned "
        f"({s['aligned_changed']} with changes), {s['divergence_blocks']} divergence block(s), "
        f"{s['only_in_a']} only in A, {s['only_in_b']} only in B.",
        "Steps are narrative steps. Aligned = same (run_type, name); "
        "token/latency deltas are B minus A.",
    ]
    lines += [f"Note: {n}" for n in data["notes"]]
    if s["identical"]:
        lines += ["", "No divergence: same step sequence, args, status, and errors on both sides."]

    unchanged: list[dict[str, Any]] = []

    def flush() -> None:
        if not unchanged:
            return
        first, last = unchanged[0], unchanged[-1]
        lines.append(
            f"- A#{first['a']['step']}-{last['a']['step']} = B#{first['b']['step']}-"
            f"{last['b']['step']}: {len(unchanged)} aligned step(s), unchanged"
        )
        unchanged.clear()

    lines += ["", "## Alignment", ""]
    for seg in data["segments"]:
        if seg["kind"] == "aligned" and not seg["changes"] and not full:
            unchanged.append(seg)
            continue
        flush()
        if seg["kind"] == "aligned":
            a, b = seg["a"], seg["b"]
            mark = "changed" if seg["changes"] else "unchanged"
            lines.append(
                f"- A#{a['step']} = B#{b['step']} [{a['run_type']}] {a['name']} — {mark} "
                f"(ids A={a['run_id']} B={b['run_id']}; tokens "
                f"{_fmt_delta(seg['token_delta'], '')}, latency "
                f"{_fmt_delta(seg['latency_delta_s'], 's')})"
            )
            for key, change in seg["changes"].items():
                lines.append(f"  - {key}: A={change['a']} | B={change['b']}")
        else:
            a_steps = [r["step"] for r in seg["a"]]
            b_steps = [r["step"] for r in seg["b"]]
            a_span = f"A#{a_steps[0]}-{a_steps[-1]}" if a_steps else "A: none"
            b_span = f"B#{b_steps[0]}-{b_steps[-1]}" if b_steps else "B: none"
            lines += ["", f"### Divergence {a_span} vs {b_span}", ""]
            lines += _block_lines("A", seg["a"], trace_a, full)
            lines += _block_lines("B", seg["b"], trace_b, full)
            lines.append("")
    flush()

    lines += [
        "",
        f"Drill down: `self-improve run-detail {trace_a} <run_id>` (or {trace_b}), "
        "`self-improve narrative <trace_id> --around-step N`.",
    ]
    return "\n".join(lines) + "\n"
