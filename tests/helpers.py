"""Helpers for building canonical Run objects in tests."""

from __future__ import annotations

from self_improve_cli.domain import Message, Run, RunType, ToolCall


def make_run(
    idx: str,
    run_type: RunType = RunType.LLM,
    name: str = "ChatModel",
    parent: str | None = "root",
    trace_id: str = "t1",
    prompt_tokens: int | None = None,
    total_tokens: int | None = None,
    inputs: dict | None = None,
    outputs: dict | None = None,
    input_messages: list[Message] | None = None,
    output_message: Message | None = None,
    error: str | None = None,
    extra: dict | None = None,
) -> Run:
    """Build a canonical Run with sensible defaults for tests."""
    dotted = f"0001root.{idx}run-{idx}"
    if parent is None:
        dotted = f"0001run-{idx}"
    return Run(
        id=f"run-{idx}",
        trace_id=trace_id,
        run_type=run_type,
        name=name,
        parent_run_id=parent,
        dotted_order=dotted,
        status="success",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
        total_tokens=total_tokens
        if total_tokens is not None
        else (prompt_tokens + 10 if prompt_tokens else None),
        prompt_tokens=prompt_tokens,
        completion_tokens=10 if prompt_tokens else None,
        error=error,
        inputs=inputs or {},
        outputs=outputs or {},
        input_messages=input_messages or [],
        output_message=output_message,
        extra=extra or {},
    )


def make_root(trace_id: str = "t1") -> Run:
    run = make_run("root", run_type=RunType.CHAIN, name="agent", parent=None, trace_id=trace_id)
    run.id = "root"
    run.dotted_order = "0001root"
    return run


def make_llm(
    idx: str, prompt_tokens: int, parent: str = "root", input_messages: list[Message] | None = None
) -> Run:
    return make_run(
        idx,
        run_type=RunType.LLM,
        prompt_tokens=prompt_tokens,
        parent=parent,
        input_messages=input_messages,
    )


def make_tool(idx: str, name: str, args: dict | None = None, parent: str = "root") -> Run:
    return make_run(idx, run_type=RunType.TOOL, name=name, parent=parent, inputs=args or {})


def make_nested_llm(idx: str, prompt_tokens: int, tool_id: str) -> Run:
    return make_run(
        idx, run_type=RunType.LLM, name="ChatAnthropic", parent=tool_id, prompt_tokens=prompt_tokens
    )


def make_msg(
    role: str,
    text: str = "",
    tool_calls: list[ToolCall] | None = None,
    tool_call_id: str | None = None,
) -> Message:
    return Message(role=role, text=text, tool_calls=tool_calls or [], tool_call_id=tool_call_id)
