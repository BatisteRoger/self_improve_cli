"""Token-efficient representations (TER) of traces.

Pure functions over canonical Run objects. No SDK dependency, no LLM calls:
deterministic and recomputable from raw/sanitized data.

Granularity levels:
- L1 skeleton:  one line per significant run.
- L2 narrative: chronological story, message deltas per LLM call.
- L3 run detail: full context of one run, on demand.

Adapted from an internal prototype, reworked around the canonical domain model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from self_improve_cli.domain import Message, Run, RunType

logger = logging.getLogger(__name__)

# Heuristics for significant runs (adjust per Target Agent when needed).
_NOISE_NAME_MARKERS = ("Middleware",)
_MAX_MSG_CHARS = 1500
_MAX_TOOL_CHARS = 600

# Error strings that indicate infrastructure/harness interruption, not an
# agent or code defect. When the error text contains one of these markers,
# the skeleton and narrative label it as infra-cancelled so the analyst
# doesn't misdiagnose a manual shutdown or timeout as an agent failure.
_INFRA_ERROR_MARKERS = (
    "CancelledError",
    "asyncio.exceptions.CancelledError",
    "KeyboardInterrupt",
    "TimeoutError",
    "concurrent.futures._base.TimeoutError",
)


def _is_infra_error(error: str) -> bool:
    """Return True if the error string matches a known infra/harness marker."""
    error_lower = error.lower()
    return any(marker.lower() in error_lower for marker in _INFRA_ERROR_MARKERS)


def _format_error(error: str, limit: int) -> str:
    """Format a run error, labeling infra-cancelled errors distinctly."""
    truncated = _truncate(error, limit)
    if _is_infra_error(error):
        return f" ERROR (infra-cancelled, not an agent failure): {truncated}"
    return f" ERROR: {truncated}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latency_s(run: Run) -> float | None:
    start, end = run.start_time, run.end_time
    if not (start and end):
        return None
    try:
        return (
            datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))
        ).total_seconds()
    except ValueError:
        return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[truncated, {len(text)} chars total]"


def _message_signature(msg: Message) -> tuple:
    return (msg.role, msg.text, json.dumps([tc.name for tc in msg.tool_calls], default=str))


def _format_tool_calls(tool_calls: list) -> str:
    lines = []
    for tc in tool_calls:
        args = json.dumps(tc.args, default=str, ensure_ascii=False)
        lines.append(f"-> tool_call {tc.name}({_truncate(args, _MAX_TOOL_CHARS)})")
    return "\n".join(lines)


def _tool_result_text(run: Run) -> str:
    """Best-effort extraction of a tool run's result."""
    outputs = run.outputs or {}
    output = outputs.get("output", outputs)
    if isinstance(output, dict):
        update = output.get("update")
        if isinstance(update, dict):
            for msg in update.get("messages") or []:
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg["content"])
    return json.dumps(output, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Significant-run filter
# ---------------------------------------------------------------------------


def significant_runs(runs: list[Run]) -> list[Run]:
    """Filter raw runs down to the ones that carry meaning.

    Kept: root, llm runs, tool runs, first-level graph nodes.
    Dropped: middleware wrappers and other duplicating chain runs.
    """
    runs = sorted(runs, key=lambda r: r.dotted_order or "")
    root_ids = {r.id for r in runs if not r.parent_run_id}

    kept = []
    for run in runs:
        name = run.name
        if any(marker in name for marker in _NOISE_NAME_MARKERS):
            continue
        is_root = not run.parent_run_id
        is_first_level = run.parent_run_id in root_ids
        if is_root or run.run_type in (RunType.LLM, RunType.TOOL) or is_first_level:
            kept.append(run)

    logger.info("Significant runs: kept %d of %d", len(kept), len(runs))
    return kept


# ---------------------------------------------------------------------------
# L1 — skeleton
# ---------------------------------------------------------------------------


def build_skeleton(runs: list[Run]) -> str:
    """One line per significant run: order, type, name, tokens, latency, error."""
    sig = significant_runs(runs)
    if not sig:
        return "# Skeleton\n\n(empty trace)\n"

    root = sig[0]
    lines = [
        f"# Skeleton — trace {root.trace_id}",
        "",
        f"root: {root.name} status={root.status} "
        f"total_tokens={root.total_tokens} latency={_latency_s(root)}s",
        "",
    ]
    for i, run in enumerate(sig):
        depth = len((run.dotted_order or "").split(".")) - 1
        indent = "  " * depth
        error = _format_error(str(run.error), 200) if run.error else ""
        tokens = run.total_tokens
        tokens_part = f" tokens={tokens}" if tokens else ""
        lines.append(
            f"{i:3d}. {indent}[{run.run_type.value}] {run.name}"
            f"{tokens_part} latency={_latency_s(run)}s id={run.id}{error}"
        )
    return "\n".join(lines) + "\n"


def skeleton_data(runs: list[Run]) -> dict[str, Any]:
    """Structured skeleton data for JSON output (composable contract).

    Returns a dict with trace_id, root summary, and a list of significant runs.
    Each run entry includes: index, id, run_type, name, depth, parent_run_id,
    status, total_tokens, latency_s, error.

    Unlike build_skeleton (which returns markdown), this returns structured
    data so an analyst agent can programmatically extract run IDs without
    parsing prose.
    """
    sig = significant_runs(runs)
    if not sig:
        return {"trace_id": None, "root": None, "runs": []}

    root = sig[0]
    run_list: list[dict[str, Any]] = []
    for i, run in enumerate(sig):
        depth = len((run.dotted_order or "").split(".")) - 1
        run_list.append(
            {
                "index": i,
                "id": run.id,
                "run_type": run.run_type.value,
                "name": run.name,
                "depth": depth,
                "parent_run_id": run.parent_run_id,
                "status": run.status,
                "total_tokens": run.total_tokens,
                "latency_s": _latency_s(run),
                "error": run.error,
            }
        )

    return {
        "trace_id": root.trace_id,
        "root": {
            "name": root.name,
            "status": root.status,
            "total_tokens": root.total_tokens,
            "latency_s": _latency_s(root),
        },
        "runs": run_list,
    }


# ---------------------------------------------------------------------------
# L2 — narrative
# ---------------------------------------------------------------------------


def _narrative_llm_step_full(run: Run, previous_msgs: list[Message]) -> tuple[str, list[Message]]:
    """Render one LLM call as a delta against the previous call's messages.

    Uses a common-prefix walk: messages are compared in order, and the walk
    stops at the first mismatch. Everything after the mismatch is printed in full.
    This is the original behavior — useful for humans skimming start-to-end,
    but degrades when the system message changes every step (dynamic prompts).
    """
    msgs = run.input_messages
    prev_sigs = [_message_signature(m) for m in previous_msgs]
    sigs = [_message_signature(m) for m in msgs]
    common = 0
    while common < min(len(prev_sigs), len(sigs)) and prev_sigs[common] == sigs[common]:
        common += 1

    lines = []
    if common:
        lines.append(f"(context: {common} unchanged messages, {len(msgs) - common} new)")
    for msg in msgs[common:]:
        lines.append(f"[{msg.role}] {_truncate(msg.text, _MAX_MSG_CHARS)}")
        if msg.tool_calls:
            lines.append(_format_tool_calls(msg.tool_calls))

    out = run.output_message
    lines.append("=> response:")
    if out is None:
        lines.append(
            f"(no parsed output) {_truncate(json.dumps(run.outputs, default=str), _MAX_TOOL_CHARS)}"
        )
    else:
        if out.text:
            lines.append(f"[{out.role}] {_truncate(out.text, _MAX_MSG_CHARS)}")
        if out.tool_calls:
            lines.append(_format_tool_calls(out.tool_calls))
    return "\n".join(lines), msgs


def _narrative_llm_step_compact(
    run: Run, previous_msgs: list[Message], step_num: int
) -> tuple[str, list[Message]]:
    """Render one LLM call as a per-index delta against the previous call's messages.

    For each message at index i, compare its signature against the message at
    index i in the previous step independently (no early stop on first mismatch).
    Identical messages collapse to one line; only genuinely new/changed messages
    are printed in full. This fixes the cascade where a dynamic system prompt
    causes the entire message list to be re-dumped every step.
    """
    msgs = run.input_messages
    prev_sigs = [_message_signature(m) for m in previous_msgs]
    sigs = [_message_signature(m) for m in msgs]

    lines = []
    unchanged = 0
    for i, msg in enumerate(msgs):
        if i < len(prev_sigs) and prev_sigs[i] == sigs[i]:
            unchanged += 1
            continue
        if unchanged:
            lines.append(f"({unchanged} unchanged messages)")
            unchanged = 0
        lines.append(f"[{msg.role}] {_truncate(msg.text, _MAX_MSG_CHARS)}")
        if msg.tool_calls:
            lines.append(_format_tool_calls(msg.tool_calls))
    if unchanged:
        lines.append(f"({unchanged} unchanged messages)")

    out = run.output_message
    lines.append("=> response:")
    if out is None:
        lines.append(
            f"(no parsed output) {_truncate(json.dumps(run.outputs, default=str), _MAX_TOOL_CHARS)}"
        )
    else:
        if out.text:
            lines.append(f"[{out.role}] {_truncate(out.text, _MAX_MSG_CHARS)}")
        if out.tool_calls:
            lines.append(_format_tool_calls(out.tool_calls))
    return "\n".join(lines), msgs


def build_narrative(runs: list[Run], mode: str = "compact") -> str:
    """Chronological story of the trace with message deltas per LLM call.

    Args:
        runs: canonical Run objects.
        mode: "compact" (default) — per-index diff, collapses unchanged messages
              even when earlier messages changed (e.g. dynamic system prompts).
              "full" — common-prefix diff, original behavior for humans skimming.
    """
    sig = significant_runs(runs)
    if not sig:
        return "# Narrative\n\n(empty trace)\n"

    root = sig[0]
    root_msgs = root.input_messages
    task = "\n".join(f"[{m.role}] {m.text}" for m in root_msgs) or json.dumps(
        root.inputs, default=str
    )

    sections = [
        f"# Narrative — trace {root.trace_id}",
        "",
        f"status={root.status} total_tokens={root.total_tokens} latency={_latency_s(root)}s",
        "",
        "## Task",
        "",
        task,
    ]

    step = 0
    previous_msgs: list[Message] = []
    for run in sig[1:]:
        if run.run_type == RunType.LLM:
            step += 1
            header = (
                f"## Step {step} — llm {run.name} "
                f"(tokens={run.total_tokens}, latency={_latency_s(run)}s, id={run.id})"
            )
            if mode == "full":
                body, previous_msgs = _narrative_llm_step_full(run, previous_msgs)
            else:
                body, previous_msgs = _narrative_llm_step_compact(run, previous_msgs, step)
            sections += ["", header, "", body]
        elif run.run_type == RunType.TOOL:
            step += 1
            args = json.dumps(run.inputs, default=str, ensure_ascii=False)
            sections += [
                "",
                f"## Step {step} — tool {run.name} (latency={_latency_s(run)}s, id={run.id})",
                "",
                f"args: {_truncate(args, _MAX_TOOL_CHARS)}",
                f"result: {_truncate(_tool_result_text(run), _MAX_TOOL_CHARS)}",
            ]
        else:
            step += 1
            outputs = json.dumps(run.outputs, default=str, ensure_ascii=False)
            sections += [
                "",
                f"## Step {step} — node {run.name} (latency={_latency_s(run)}s, id={run.id})",
                "",
                f"output: {_truncate(outputs, _MAX_TOOL_CHARS)}",
            ]
        if run.error:
            sections.append(_format_error(str(run.error), 1000))

    return "\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# L3 — run detail
# ---------------------------------------------------------------------------


def run_detail(runs: list[Run], run_id: str) -> str:
    """Full untruncated context of one run, reconstructed from canonical data."""
    matches = [r for r in runs if str(r.id) == str(run_id)]
    if not matches:
        # Recoverable error: suggest how to find valid run IDs.
        available = sorted(
            f"  {r.id}  {r.run_type.value}  {r.name}" for r in significant_runs(runs)
        )
        hint = "Run `self-improve skeleton <trace_id>` for a compact overview."
        if available:
            sample = "\n".join(available[:10])
            if len(available) > 10:
                sample += f"\n  ... ({len(available) - 10} more)"
            raise ValueError(
                f"Run {run_id} not found in trace.\n\n"
                f"Available significant runs:\n{sample}\n\n{hint}"
            )
        raise ValueError(f"Run {run_id} not found in trace.\n\n{hint}")
    run = matches[0]

    # Find parent and child runs for navigation (connectedness).
    parent = None
    if run.parent_run_id:
        parent = next((r for r in runs if r.id == run.parent_run_id), None)
    children = [r for r in runs if r.parent_run_id == run.id]

    lines = [
        f"# Run detail — {run.name} ({run.run_type.value}) id={run.id}",
        "",
        f"status={run.status} tokens={run.total_tokens} "
        f"latency={_latency_s(run)}s error={run.error}",
        "",
    ]

    # Navigation references (connectedness contract).
    if parent or children:
        lines.append("## Related runs")
        lines.append("")
        if parent:
            lines.append(
                f"Parent: {parent.name} ({parent.run_type.value}) id={parent.id}  "
                f"→ `self-improve run-detail <trace_id> {parent.id}`"
            )
        for child in children:
            lines.append(
                f"Child:  {child.name} ({child.run_type.value}) id={child.id}  "
                f"→ `self-improve run-detail <trace_id> {child.id}`"
            )
        lines.append("")
    if run.run_type == RunType.LLM:
        lines.append("## Input messages")
        for msg in run.input_messages:
            lines += ["", f"### [{msg.role}]", "", msg.text]
            if msg.tool_calls:
                lines.append(_format_tool_calls(msg.tool_calls))
        out = run.output_message
        lines += ["", "## Output", ""]
        if out:
            lines.append(out.text)
            if out.tool_calls:
                lines.append(_format_tool_calls(out.tool_calls))
        else:
            lines.append(json.dumps(run.outputs, indent=2, default=str))
    else:
        lines += [
            "## Inputs",
            "",
            json.dumps(run.inputs, indent=2, default=str, ensure_ascii=False),
            "",
            "## Outputs",
            "",
            json.dumps(run.outputs, indent=2, default=str, ensure_ascii=False),
        ]
    return "\n".join(lines) + "\n"


def run_detail_data(runs: list[Run], run_id: str) -> dict[str, Any]:
    """Structured run detail for JSON output (composable contract).

    Returns a dict with run metadata, related runs (parent/children), and
    either input_messages+output (for LLM runs) or inputs+outputs (for other
    runs). Tool calls are included as structured objects, not prose.

    Raises ValueError if the run is not found (same recovery as run_detail).
    """
    matches = [r for r in runs if str(r.id) == str(run_id)]
    if not matches:
        available = sorted(
            f"  {r.id}  {r.run_type.value}  {r.name}" for r in significant_runs(runs)
        )
        hint = "Run `self-improve skeleton <trace_id>` for a compact overview."
        if available:
            sample = "\n".join(available[:10])
            if len(available) > 10:
                sample += f"\n  ... ({len(available) - 10} more)"
            raise ValueError(
                f"Run {run_id} not found in trace.\n\n"
                f"Available significant runs:\n{sample}\n\n{hint}"
            )
        raise ValueError(f"Run {run_id} not found in trace.\n\n{hint}")
    run = matches[0]

    parent = None
    if run.parent_run_id:
        parent_run = next((r for r in runs if r.id == run.parent_run_id), None)
        if parent_run:
            parent = {
                "id": parent_run.id,
                "name": parent_run.name,
                "run_type": parent_run.run_type.value,
            }
    children = [
        {"id": r.id, "name": r.name, "run_type": r.run_type.value}
        for r in runs
        if r.parent_run_id == run.id
    ]

    result: dict[str, Any] = {
        "id": run.id,
        "trace_id": run.trace_id,
        "run_type": run.run_type.value,
        "name": run.name,
        "status": run.status,
        "total_tokens": run.total_tokens,
        "latency_s": _latency_s(run),
        "error": run.error,
        "parent": parent,
        "children": children,
    }

    if run.run_type == RunType.LLM:
        result["input_messages"] = [
            {
                "role": msg.role,
                "text": msg.text,
                "tool_calls": [
                    {"name": tc.name, "args": tc.args, "id": tc.id}
                    for tc in msg.tool_calls
                ],
                "tool_call_id": msg.tool_call_id,
            }
            for msg in run.input_messages
        ]
        out = run.output_message
        if out:
            result["output"] = {
                "role": out.role,
                "text": out.text,
                "tool_calls": [
                    {"name": tc.name, "args": tc.args, "id": tc.id}
                    for tc in out.tool_calls
                ],
            }
        else:
            result["output"] = run.outputs
    else:
        result["inputs"] = run.inputs
        result["outputs"] = run.outputs

    return result


# ---------------------------------------------------------------------------
# L1 — tools overview & detail
# ---------------------------------------------------------------------------

# Scope labels inferred from tool-set signatures. Keys are frozensets of tool
# names; values are short human-readable labels. This is heuristic — the goal
# is to help the analyst, not to be exhaustive.
_SCOPE_LABELS: dict[frozenset[str], str] = {
    frozenset(): "guardrails (no tools)",
}

# Fallback inference when no explicit label matches. We look for known
# tool-name patterns to guess the scope.
_SCOPE_PATTERNS: list[tuple[frozenset[str], str]] = [
    (frozenset({"search_agent", "financial_agent"}), "supervisor"),
    (frozenset({"search_agent"}), "supervisor (search only)"),
    (frozenset({"rag_tool", "web_search_tool_safe"}), "search_agent"),
]


def _extract_tools(run: Run) -> list[dict[str, Any]]:
    """Extract the tool definitions passed to an LLM run.

    LangSmith stores these in extra.extra.invocation_params.tools as a list of
    {"type": "function", "function": {"name", "description", "parameters"}}.
    Returns an empty list if the run has no tools or the path is missing.
    """
    extra = run.extra.get("extra") or {}
    invocation_params = extra.get("invocation_params") or {}
    tools = invocation_params.get("tools") or []
    return tools if isinstance(tools, list) else []


def _tool_names(tools: list[dict[str, Any]]) -> frozenset[str]:
    """Extract a frozenset of tool names from a tool definition list."""
    names = []
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function") or t
            name = fn.get("name")
            if name:
                names.append(name)
    return frozenset(names)


def _infer_scope_label(tool_set: frozenset[str]) -> str:
    """Infer a short human-readable label for a tool-set signature."""
    if tool_set in _SCOPE_LABELS:
        return _SCOPE_LABELS[tool_set]
    for pattern, label in _SCOPE_PATTERNS:
        if pattern.issubset(tool_set):
            return label
    # Fallback: list the first few tool names
    sample = sorted(tool_set)[:3]
    return ", ".join(sample) + ("…" if len(tool_set) > 3 else "")


def build_tools_overview(runs: list[Run]) -> str:
    """Compact matrix of tools available per LLM run, grouped by scope.

    LLM runs are grouped by their tool-set signature. Each group becomes a
    column in the matrix. Tools are rows. An ``X`` marks presence.
    """
    llm_runs = [r for r in runs if r.run_type == RunType.LLM]
    if not llm_runs:
        return "# Tools\n\n(no LLM runs in this trace)\n"

    # Group runs by tool-set signature
    groups: dict[frozenset[str], list[Run]] = {}
    for run in llm_runs:
        tools = _extract_tools(run)
        sig = _tool_names(tools)
        groups.setdefault(sig, []).append(run)

    # Order groups by first appearance (stable)
    ordered_sigs = list(groups.keys())

    # Collect all unique tool names across groups, preserving first-seen order
    all_tools: list[str] = []
    seen: set[str] = set()
    for sig in ordered_sigs:
        for name in sorted(sig):
            if name not in seen:
                all_tools.append(name)
                seen.add(name)

    # Build header
    root = runs[0] if runs else llm_runs[0]
    lines = [
        f"# Tools — trace {root.trace_id}",
        "",
        f"{len(llm_runs)} LLM runs, {len(ordered_sigs)} scope(s), {len(all_tools)} unique tool(s)",
        "",
    ]

    # Column headers: scope label + run count
    col_labels = []
    for sig in ordered_sigs:
        label = _infer_scope_label(sig)
        count = len(groups[sig])
        col_labels.append(f"{label} ({count} run{'s' if count > 1 else ''})")

    # Compute column width (min 4 for "tool")
    col_width = max(len("tool"), max(len(c) for c in col_labels)) if col_labels else 4

    # Header row
    header = f"{'tool':<{col_width}}  " + "  ".join(f"{c:^{col_width}}" for c in col_labels)
    lines.append(header)
    lines.append("-" * len(header))

    # Matrix rows
    for tool_name in all_tools:
        row = f"{tool_name:<{col_width}}  "
        marks = []
        for sig in ordered_sigs:
            mark = "X" if tool_name in sig else " "
            marks.append(f"{mark:^{col_width}}")
        row += "  ".join(marks)
        lines.append(row)

    # Footer: run IDs per scope
    lines.append("")
    lines.append("## Runs per scope")
    lines.append("")
    for sig in ordered_sigs:
        label = _infer_scope_label(sig)
        run_ids = [r.id for r in groups[sig]]
        lines.append(f"**{label}** ({len(run_ids)} run{'s' if len(run_ids) > 1 else ''}):")
        for rid in run_ids:
            lines.append(f"  - {rid}")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_tools_detail(runs: list[Run], tool_name: str | None = None) -> str:
    """Full docstrings and parameter schemas for tools available in the trace.

    Args:
        runs: canonical Run objects.
        tool_name: if given, show only that tool. If None, show all unique tools.
    """
    llm_runs = [r for r in runs if r.run_type == RunType.LLM]
    if not llm_runs:
        return "# Tools (detail)\n\n(no LLM runs in this trace)\n"

    # Collect unique tool definitions: name -> (definition, scope_labels)
    tools_by_name: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for run in llm_runs:
        tools = _extract_tools(run)
        sig = _tool_names(tools)
        scope_label = _infer_scope_label(sig)
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or t
            name = fn.get("name")
            if not name:
                continue
            if name not in tools_by_name:
                tools_by_name[name] = (fn, [])
            if scope_label not in tools_by_name[name][1]:
                tools_by_name[name][1].append(scope_label)

    if tool_name and tool_name not in tools_by_name:
        available = sorted(tools_by_name.keys())
        return (
            f"# Tools (detail)\n\n"
            f"Tool '{tool_name}' not found in this trace.\n"
            f"Available: {', '.join(available)}\n"
        )

    root = runs[0] if runs else llm_runs[0]
    selected = [tool_name] if tool_name else sorted(tools_by_name.keys())

    lines = [f"# Tools (detail) — trace {root.trace_id}", ""]

    for name in selected:
        fn, scopes = tools_by_name[name]
        desc = fn.get("description", "(no description)")
        params = fn.get("parameters", {})

        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"**Scopes:** {', '.join(scopes)}")
        lines.append("")
        lines.append("### Description")
        lines.append("")
        lines.append(desc)
        lines.append("")

        if params and isinstance(params, dict) and params.get("properties"):
            lines.append("### Parameters")
            lines.append("")
            props = params.get("properties", {})
            required = set(params.get("required", []))
            for pname, pschema in props.items():
                req_marker = " (required)" if pname in required else ""
                ptype = pschema.get("type", "?") if isinstance(pschema, dict) else "?"
                pdesc = pschema.get("description", "") if isinstance(pschema, dict) else ""
                lines.append(f"- **{pname}** (`{ptype}`{req_marker}): {pdesc}")
            lines.append("")
        elif params and isinstance(params, dict):
            lines.append("### Parameters")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(params, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_ter(trace_id: str, runs: list[Run], store: Any) -> dict[str, Any]:
    """Build and write all TER files for a trace.

    Args:
        trace_id: The trace ID.
        runs: Canonical Run objects (sanitized or raw).
        store: A TraceStore instance for persistence.

    Returns a small stats dict (sizes and compression ratio).
    """
    from self_improve_cli.metrics.context_metrics import build_context_metrics
    from self_improve_cli.metrics.tool_metrics import build_tool_metrics

    skeleton = build_skeleton(runs)
    narrative_compact = build_narrative(runs, mode="compact")
    narrative_full = build_narrative(runs, mode="full")
    tool_metrics = build_tool_metrics(runs)
    context_metrics = build_context_metrics(runs)

    store.save_ter_file(trace_id, "skeleton.md", skeleton)
    store.save_ter_file(trace_id, "narrative_compact.md", narrative_compact)
    store.save_ter_file(trace_id, "narrative_full.md", narrative_full)
    store.save_ter_file(trace_id, "tool_metrics.md", tool_metrics)
    store.save_ter_file(trace_id, "context_metrics.md", context_metrics)

    # Compression ratio: sanitized trace size vs compact narrative size.
    sanitized_path = store.traces_dir / trace_id / "sanitized.json"
    raw_chars = sanitized_path.stat().st_size if sanitized_path.exists() else 0
    compression_ratio = round(raw_chars / max(len(narrative_compact), 1), 1) if raw_chars else 0.0

    stats = {
        "trace_id": trace_id,
        "sanitized_chars": raw_chars,
        "skeleton_chars": len(skeleton),
        "narrative_compact_chars": len(narrative_compact),
        "narrative_full_chars": len(narrative_full),
        "narrative_chars": len(narrative_compact),
        "narrative_est_tokens": len(narrative_compact) // 4,
        "compression_ratio": compression_ratio,
        "tool_metrics_chars": len(tool_metrics),
        "context_metrics_chars": len(context_metrics),
    }
    logger.info(
        "TER written for trace %s — narrative_compact=%.1fKB (~%d tokens, %.1fx smaller) "
        "narrative_full=%.1fKB skeleton=%.1fKB tool_metrics=%.1fKB context_metrics=%.1fKB",
        trace_id,
        len(narrative_compact) / 1024,
        stats["narrative_est_tokens"],
        compression_ratio,
        len(narrative_full) / 1024,
        len(skeleton) / 1024,
        len(tool_metrics) / 1024,
        len(context_metrics) / 1024,
    )
    return stats
