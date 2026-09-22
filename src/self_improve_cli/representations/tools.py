"""L1 tools — per-scope tool availability matrix and tool detail view."""

from __future__ import annotations

import json
from typing import Any

from self_improve_cli.domain import Run, RunType

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
