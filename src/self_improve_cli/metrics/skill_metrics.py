"""Skill invocation metrics — detect and cost skill reads in a trace.

Pure functions over canonical Run objects. No SDK dependency, no LLM calls:
deterministic and recomputable from sanitized data.

A "skill invocation" is a tool call (typically read_file or grep) whose target
path contains a SKILL.md file. This is the pattern used by agent frameworks
that load skills via a filesystem backend (e.g. deepagents).

The metrics are raw facts (which skill, when, how many tokens it added).
Interpreting them ("skill was useful", "skill caused context bloat") is the
job of the Analyst Agent that consumes this file, not of this module.

APPROXIMATE: token deltas assume the skill is the only cause of context
growth between two LLM steps. Latency deltas are wall-clock and include
the tool call itself plus any downstream effect.
"""

from __future__ import annotations

import json
from typing import Any

from self_improve_cli.domain import Run, RunType
from self_improve_cli.metrics.tool_metrics import _tool_args, extract_target_key
from self_improve_cli.representations import significant_runs

# --- Configurable heuristics ------------------------------------------------

_SKILL_FILE_PATTERN = "SKILL.md"
_SKILL_TOOL_NAMES = {"read_file", "grep"}
_PATH_KEYS = ("file_path", "path", "filename", "file", "uri", "url")


# --- Helpers ----------------------------------------------------------------


def _extract_skill_path(run: Run) -> str | None:
    """Extract the file path from a skill tool call.

    Handles two arg layouts:
    1. Direct: ``{"file_path": "/skill/SKILL.md"}``
    2. Nested JSON string: ``{"input": "{\\"file_path\\": \\"/skill/SKILL.md\\"}"}``

    Returns the path string or None.
    """
    args = _tool_args(run)

    # 1. Try direct path keys
    for key in _PATH_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val:
            return val

    # 2. Try parsing a JSON string under common wrapper keys
    for wrapper in ("input", "args", "inputs"):
        raw = args.get(wrapper)
        if isinstance(raw, str) and raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    for key in _PATH_KEYS:
                        val = parsed.get(key)
                        if isinstance(val, str) and val:
                            return val
            except (json.JSONDecodeError, TypeError):
                pass

    # 3. Fallback to extract_target_key (heuristic short-identifier match)
    target = extract_target_key(run.name, args)
    if target and _SKILL_FILE_PATTERN in target:
        return target

    return None


# --- 1. Skill detection -----------------------------------------------------


def extract_skill_name(file_path: str) -> str | None:
    """Extract a skill name from a file path.

    A skill path looks like ``/priorisation-financiere/SKILL.md`` — the skill
    name is the directory immediately above the SKILL.md file. Returns None if
    the path does not contain a SKILL.md file.
    """
    if _SKILL_FILE_PATTERN not in file_path:
        return None
    # Normalize separators and split
    normalized = file_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    # Find the part that is exactly SKILL.md
    idx = None
    for i, part in enumerate(parts):
        if part == _SKILL_FILE_PATTERN:
            idx = i
            break
    if idx is None or idx == 0:
        return None
    return parts[idx - 1]


def _is_skill_tool_call(run: Run) -> bool:
    """Check if a tool run is a skill invocation (read_file/grep on a SKILL.md)."""
    if run.run_type != RunType.TOOL:
        return False
    if run.name not in _SKILL_TOOL_NAMES:
        return False
    path = _extract_skill_path(run)
    if path is None:
        return False
    return _SKILL_FILE_PATTERN in path


def skill_invocations(runs: list[Run]) -> list[dict[str, Any]]:
    """Detect all skill invocations in a trace.

    Returns one dict per invocation:
    ``{skill_name, tool_name, file_path, step_index, run_id}``.

    ``step_index`` is the 0-based position of the tool run among significant
    runs, useful for correlating with LLM steps in other metrics.
    """
    sig = significant_runs(runs)
    results: list[dict[str, Any]] = []
    for i, run in enumerate(sig):
        if not _is_skill_tool_call(run):
            continue
        path = _extract_skill_path(run) or ""
        skill_name = extract_skill_name(path)
        results.append(
            {
                "skill_name": skill_name,
                "tool_name": run.name,
                "file_path": path,
                "step_index": i,
                "run_id": run.id,
            }
        )
    return results


# --- 2. Skill token cost (approximate) --------------------------------------


def _context_tokens(run: Run) -> int:
    """Context size submitted to an LLM call.

    Prefers prompt_tokens (the accumulated context) over total_tokens, which
    also includes completion_tokens and is therefore noisier.
    """
    return run.prompt_tokens or run.total_tokens or 0


def _parse_latency_ms(run: Run) -> int | None:
    """Estimate wall-clock latency of a run in milliseconds.

    APPROXIMATE: depends on start/end time format and precision.
    Returns None if times are missing or unparseable.
    """
    if not run.start_time or not run.end_time:
        return None
    try:
        from datetime import datetime

        # Handle ISO format with optional timezone
        start = datetime.fromisoformat(run.start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(run.end_time.replace("Z", "+00:00"))
        return int((end - start).total_seconds() * 1000)
    except (ValueError, TypeError):
        return None


def skill_token_cost(runs: list[Run]) -> list[dict[str, Any]]:
    """Estimate the marginal token and latency cost of each skill invocation.

    For each skill invocation, compares the LLM step immediately before and
    after the tool call:

    - ``token_delta``: prompt_tokens after - prompt_tokens before
    - ``latency_delta_ms``: end_time after - start_time before (wall-clock)
    - ``extra_llm_call``: True if the skill caused an additional LLM call

    APPROXIMATE: assumes the skill is the only cause of the delta. In
    practice, other tool calls between two LLM steps also contribute. The
    Analyst Agent should cross-reference with tool_metrics for context.

    Returns one dict per skill invocation:
    ``{skill_name, token_delta, latency_delta_ms, extra_llm_call}``.
    """
    sig = significant_runs(runs)
    invocations = skill_invocations(runs)

    if not invocations:
        return []

    # Build a lookup: run_id -> index in sig
    run_index = {r.id: i for i, r in enumerate(sig)}

    results: list[dict[str, Any]] = []
    for inv in invocations:
        tool_idx = run_index.get(inv["run_id"])
        if tool_idx is None:
            continue

        # Find the LLM step immediately before the tool call
        prev_llm_tokens: int | None = None
        for j in range(tool_idx - 1, -1, -1):
            if sig[j].run_type == RunType.LLM:
                prev_llm_tokens = _context_tokens(sig[j])
                break

        # Find the LLM step immediately after the tool call
        next_llm_tokens: int | None = None
        next_llm_latency: int | None = None
        for j in range(tool_idx + 1, len(sig)):
            if sig[j].run_type == RunType.LLM:
                next_llm_tokens = _context_tokens(sig[j])
                next_llm_latency = _parse_latency_ms(sig[j])
                break

        token_delta = None
        if prev_llm_tokens is not None and next_llm_tokens is not None:
            token_delta = next_llm_tokens - prev_llm_tokens

        # Latency delta: from the previous LLM start to the next LLM end
        # This captures the full round-trip cost of the skill read
        latency_delta_ms = None
        if next_llm_latency is not None:
            # Use the next LLM run's own latency as a proxy for the marginal cost
            # (the tool call + the additional LLM call to process the skill)
            latency_delta_ms = next_llm_latency

        # An extra LLM call occurred if there are more LLM runs after than before
        # In a simple trace: LLM -> tool -> LLM (2 LLM calls vs 1 without skill)
        extra_llm_call = next_llm_tokens is not None and prev_llm_tokens is not None

        results.append(
            {
                "skill_name": inv["skill_name"],
                "token_delta": token_delta,
                "latency_delta_ms": latency_delta_ms,
                "extra_llm_call": extra_llm_call,
            }
        )
    return results


# --- Markdown assembly ------------------------------------------------------


def build_skill_metrics(runs: list[Run]) -> str:
    """Assemble skill_metrics.md from canonical runs."""
    sig = significant_runs(runs)
    if not sig:
        return "# Skill Metrics\n\n(empty trace)\n"

    trace_id = sig[0].trace_id
    sections = [f"# Skill Metrics — trace {trace_id}", ""]

    invocations = skill_invocations(runs)
    if not invocations:
        sections.append("No skill invocations detected in this trace.")
        sections.append("")
        return "\n".join(sections) + "\n"

    sections += ["## Invocations", ""]
    for inv in invocations:
        sections.append(
            f"- {inv['skill_name'] or '(unknown)'} "
            f"via {inv['tool_name']} "
            f"at step {inv['step_index']} "
            f"({inv['file_path']})"
        )
    sections.append("")

    costs = skill_token_cost(runs)
    if costs:
        sections += ["## Token cost (approximate)", ""]
        sections.append(
            "Note: token deltas assume the skill is the only cause of context "
            "growth between LLM steps. Cross-reference with tool_metrics."
        )
        sections.append("")
        for cost in costs:
            delta_str = f"+{cost['token_delta']:,}" if cost["token_delta"] is not None else "N/A"
            latency_str = (
                f"{cost['latency_delta_ms']}ms" if cost["latency_delta_ms"] is not None else "N/A"
            )
            extra_str = "yes" if cost["extra_llm_call"] else "no"
            sections.append(
                f"- {cost['skill_name'] or '(unknown)'}: "
                f"tokens {delta_str}, "
                f"latency {latency_str}, "
                f"extra LLM call: {extra_str}"
            )
        sections.append("")

    return "\n".join(sections) + "\n"
