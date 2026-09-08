"""Synthetic benchmark trace builders.

Each function returns a list of canonical Run objects representing a trace
with a known issue. All content is synthetic — no real data.

The builders use the test helpers (make_root, make_run, etc.) for consistency
with the rest of the test suite.
"""

from __future__ import annotations

from collections.abc import Callable

from self_improve_cli.domain import Message, Run, RunType, ToolCall
from tests.helpers import make_msg, make_root, make_run


def _make_trace(trace_id: str, runs: list[Run]) -> list[Run]:
    """Set trace_id on all runs and return them."""
    for r in runs:
        r.trace_id = trace_id
    return runs


# ---------------------------------------------------------------------------
# 1. Ignored tool error
# ---------------------------------------------------------------------------


def build_ignored_tool_error() -> list[Run]:
    """Agent calls a tool that returns a 403, then continues without adapting.

    The agent's next LLM call does not mention the error and proceeds as if
    the tool succeeded. The error IS visible in the tool run's output, but
    the agent's response ignores it.
    """
    root = make_root(trace_id="ignored_tool_error")
    root.inputs = {"messages": [{"type": "human", "content": "Delete the temp files in /tmp."}]}
    root.outputs = {"messages": [{"type": "ai", "content": "Done."}]}

    llm1 = make_run(
        "llm-1",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="ignored_tool_error",
        total_tokens=100,
        prompt_tokens=80,
    )
    llm1.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Delete the temp files in /tmp."),
    ]
    llm1.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="delete_files", args={"path": "/tmp"}, id="call-1")],
    )

    tool1 = make_run(
        "tool-1",
        run_type=RunType.TOOL,
        name="delete_files",
        parent="root",
        trace_id="ignored_tool_error",
    )
    tool1.inputs = {"path": "/tmp"}
    tool1.outputs = {"error": "403 Forbidden: permission denied"}
    tool1.error = "403 Forbidden: permission denied"
    tool1.status = "error"

    llm2 = make_run(
        "llm-2",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="ignored_tool_error",
        total_tokens=120,
        prompt_tokens=100,
    )
    llm2.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Delete the temp files in /tmp."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="delete_files", args={"path": "/tmp"}, id="call-1")],
        ),
        Message(role="tool", text="403 Forbidden: permission denied", tool_call_id="call-1"),
    ]
    llm2.output_message = Message(role="ai", text="Done! I've deleted the temp files in /tmp.")

    return _make_trace("ignored_tool_error", [root, llm1, tool1, llm2])


# ---------------------------------------------------------------------------
# 2. Unverified completion
# ---------------------------------------------------------------------------


def build_unverified_completion() -> list[Run]:
    """Agent edits a file and claims success without verifying the result.

    The agent's final message says "Done!" but no tool call after the edit
    checks the result (no read_file, no test run, no diff).
    """
    root = make_root(trace_id="unverified_completion")
    root.inputs = {"messages": [{"type": "human", "content": "Fix the typo in config.py."}]}
    root.outputs = {"messages": [{"type": "ai", "content": "Done!"}]}

    llm1 = make_run(
        "llm-1",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="unverified_completion",
        total_tokens=100,
        prompt_tokens=80,
    )
    llm1.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Fix the typo in config.py."),
    ]
    llm1.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="edit_file", args={"path": "config.py"}, id="call-1")],
    )

    tool1 = make_run(
        "tool-1",
        run_type=RunType.TOOL,
        name="edit_file",
        parent="root",
        trace_id="unverified_completion",
    )
    tool1.inputs = {"path": "config.py", "old_string": "recieve", "new_string": "receive"}
    tool1.outputs = {"success": True}

    llm2 = make_run(
        "llm-2",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="unverified_completion",
        total_tokens=110,
        prompt_tokens=90,
    )
    llm2.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Fix the typo in config.py."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="edit_file", args={"path": "config.py"}, id="call-1")],
        ),
        Message(role="tool", text="success: true", tool_call_id="call-1"),
    ]
    llm2.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="edit_file", args={"path": "config.py"}, id="call-2")],
    )

    tool2 = make_run(
        "tool-2",
        run_type=RunType.TOOL,
        name="edit_file",
        parent="root",
        trace_id="unverified_completion",
    )
    tool2.inputs = {"path": "config.py", "old_string": "recieve", "new_string": "receive"}
    tool2.outputs = {"success": True}

    llm3 = make_run(
        "llm-3",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="unverified_completion",
        total_tokens=120,
        prompt_tokens=100,
    )
    llm3.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Fix the typo in config.py."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="edit_file", args={"path": "config.py"}, id="call-1")],
        ),
        Message(role="tool", text="success: true", tool_call_id="call-1"),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="edit_file", args={"path": "config.py"}, id="call-2")],
        ),
        Message(role="tool", text="success: true", tool_call_id="call-2"),
    ]
    llm3.output_message = Message(role="ai", text="Done! I've fixed the typo in config.py.")

    return _make_trace("unverified_completion", [root, llm1, tool1, llm2, tool2, llm3])


# ---------------------------------------------------------------------------
# 3. Legitimate repetition (false-positive guard)
# ---------------------------------------------------------------------------


def build_legitimate_repetition() -> list[Run]:
    """Agent reads a file twice because it changed between reads.

    The first read returns version A. An edit happens. The second read
    returns version B. The repetition is legitimate — the file changed.
    An analyst should NOT flag this as redundant work.
    """
    root = make_root(trace_id="legitimate_repetition")
    root.inputs = {
        "messages": [{"type": "human", "content": "Check config.py and update the port."}]
    }
    root.outputs = {"messages": [{"type": "ai", "content": "Done."}]}

    llm1 = make_run(
        "llm-1",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="legitimate_repetition",
        total_tokens=100,
        prompt_tokens=80,
    )
    llm1.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Check config.py and update the port."),
    ]
    llm1.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="read_file", args={"path": "config.py"}, id="call-1")],
    )

    tool1 = make_run(
        "tool-1",
        run_type=RunType.TOOL,
        name="read_file",
        parent="root",
        trace_id="legitimate_repetition",
    )
    tool1.inputs = {"path": "config.py"}
    tool1.outputs = {"content": "port = 8080\ndebug = false"}

    llm2 = make_run(
        "llm-2",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="legitimate_repetition",
        total_tokens=110,
        prompt_tokens=90,
    )
    llm2.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Check config.py and update the port."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="read_file", args={"path": "config.py"}, id="call-1")],
        ),
        Message(role="tool", text="port = 8080\ndebug = false", tool_call_id="call-1"),
    ]
    llm2.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="edit_file", args={"path": "config.py"}, id="call-2")],
    )

    tool2 = make_run(
        "tool-2",
        run_type=RunType.TOOL,
        name="edit_file",
        parent="root",
        trace_id="legitimate_repetition",
    )
    tool2.inputs = {"path": "config.py", "old_string": "port = 8080", "new_string": "port = 3000"}
    tool2.outputs = {"success": True}

    llm3 = make_run(
        "llm-3",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="legitimate_repetition",
        total_tokens=120,
        prompt_tokens=100,
    )
    llm3.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Check config.py and update the port."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="read_file", args={"path": "config.py"}, id="call-1")],
        ),
        Message(role="tool", text="port = 8080\ndebug = false", tool_call_id="call-1"),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="edit_file", args={"path": "config.py"}, id="call-2")],
        ),
        Message(role="tool", text="success: true", tool_call_id="call-2"),
    ]
    llm3.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="read_file", args={"path": "config.py"}, id="call-3")],
    )

    tool3 = make_run(
        "tool-3",
        run_type=RunType.TOOL,
        name="read_file",
        parent="root",
        trace_id="legitimate_repetition",
    )
    tool3.inputs = {"path": "config.py"}
    tool3.outputs = {"content": "port = 3000\ndebug = false"}

    llm4 = make_run(
        "llm-4",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="legitimate_repetition",
        total_tokens=130,
        prompt_tokens=110,
    )
    llm4.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Check config.py and update the port."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="read_file", args={"path": "config.py"}, id="call-1")],
        ),
        Message(role="tool", text="port = 8080\ndebug = false", tool_call_id="call-1"),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="edit_file", args={"path": "config.py"}, id="call-2")],
        ),
        Message(role="tool", text="success: true", tool_call_id="call-2"),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="read_file", args={"path": "config.py"}, id="call-3")],
        ),
        Message(role="tool", text="port = 3000\ndebug = false", tool_call_id="call-3"),
    ]
    llm4.output_message = Message(role="ai", text="Done. The port is now 3000.")

    return _make_trace("legitimate_repetition", [root, llm1, tool1, llm2, tool2, llm3, tool3, llm4])


# ---------------------------------------------------------------------------
# 4. Lost constraint after compaction
# ---------------------------------------------------------------------------


def build_lost_constraint_after_compaction() -> list[Run]:
    """A user constraint disappears from the context and the agent violates it.

    Step 1: user says "don't use the network." The agent acknowledges.
    Step 2: (simulated compaction — the constraint is gone from the context)
    Step 3: the agent calls a network tool, violating the constraint.
    """
    root = make_root(trace_id="lost_constraint_after_compaction")
    root.inputs = {
        "messages": [
            {"type": "human", "content": "Fetch the data without using the network."}
        ]
    }
    root.outputs = {"messages": [{"type": "ai", "content": "Done."}]}

    llm1 = make_run(
        "llm-1",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="lost_constraint_after_compaction",
        total_tokens=100,
        prompt_tokens=80,
    )
    llm1.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Fetch the data without using the network."),
    ]
    llm1.output_message = Message(
        role="ai",
        text="I'll fetch the data from the local cache without using the network.",
    )

    # Simulated compaction: the human message with the constraint is gone,
    # and the previous AI acknowledgment is summarized without the constraint.
    llm3 = make_run(
        "llm-3",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="lost_constraint_after_compaction",
        total_tokens=90,
        prompt_tokens=70,
    )
    llm3.input_messages = [
        make_msg("system", "You are a helpful agent."),
        # NOTE: the human constraint "without using the network" is absent.
        make_msg("human", "Fetch the data."),
        # NOTE: the previous AI acknowledgment is summarized without the constraint.
        Message(role="ai", text="I'll fetch the data now."),
    ]
    llm3.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="web_search", args={"query": "data"}, id="call-1")],
    )

    tool1 = make_run(
        "tool-1",
        run_type=RunType.TOOL,
        name="web_search",
        parent="root",
        trace_id="lost_constraint_after_compaction",
    )
    tool1.inputs = {"query": "data"}
    tool1.outputs = {"results": ["result1", "result2"]}

    return _make_trace("lost_constraint_after_compaction", [root, llm1, llm3, tool1])


# ---------------------------------------------------------------------------
# 5. Clean execution (false-positive control)
# ---------------------------------------------------------------------------


def build_clean_execution() -> list[Run]:
    """A successful trace that should NOT be criticized.

    The agent reads a file, makes a correct edit, verifies the result by
    reading the file again, and reports success. No errors, no ignored
    feedback, no unverified claims, no lost constraints.
    """
    root = make_root(trace_id="clean_execution")
    root.inputs = {
        "messages": [{"type": "human", "content": "Update the version in package.json."}]
    }
    root.outputs = {
        "messages": [{"type": "ai", "content": "Done. Version updated to 1.2.0."}]
    }

    llm1 = make_run(
        "llm-1",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="clean_execution",
        total_tokens=100,
        prompt_tokens=80,
    )
    llm1.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Update the version in package.json."),
    ]
    llm1.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="read_file", args={"path": "package.json"}, id="call-1")],
    )

    tool1 = make_run(
        "tool-1",
        run_type=RunType.TOOL,
        name="read_file",
        parent="root",
        trace_id="clean_execution",
    )
    tool1.inputs = {"path": "package.json"}
    tool1.outputs = {"content": '{"version": "1.1.0"}'}

    llm2 = make_run(
        "llm-2",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="clean_execution",
        total_tokens=110,
        prompt_tokens=90,
    )
    llm2.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Update the version in package.json."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="read_file", args={"path": "package.json"}, id="call-1")],
        ),
        Message(role="tool", text='{"version": "1.1.0"}', tool_call_id="call-1"),
    ]
    llm2.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="edit_file", args={"path": "package.json"}, id="call-2")],
    )

    tool2 = make_run(
        "tool-2",
        run_type=RunType.TOOL,
        name="edit_file",
        parent="root",
        trace_id="clean_execution",
    )
    tool2.inputs = {"path": "package.json", "old_string": "1.1.0", "new_string": "1.2.0"}
    tool2.outputs = {"success": True}

    llm3 = make_run(
        "llm-3",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="clean_execution",
        total_tokens=120,
        prompt_tokens=100,
    )
    llm3.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Update the version in package.json."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="read_file", args={"path": "package.json"}, id="call-1")],
        ),
        Message(role="tool", text='{"version": "1.1.0"}', tool_call_id="call-1"),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="edit_file", args={"path": "package.json"}, id="call-2")],
        ),
        Message(role="tool", text="success: true", tool_call_id="call-2"),
    ]
    llm3.output_message = Message(
        role="ai",
        text="",
        tool_calls=[ToolCall(name="read_file", args={"path": "package.json"}, id="call-3")],
    )

    tool3 = make_run(
        "tool-3",
        run_type=RunType.TOOL,
        name="read_file",
        parent="root",
        trace_id="clean_execution",
    )
    tool3.inputs = {"path": "package.json"}
    tool3.outputs = {"content": '{"version": "1.2.0"}'}

    llm4 = make_run(
        "llm-4",
        run_type=RunType.LLM,
        name="ChatModel",
        parent="root",
        trace_id="clean_execution",
        total_tokens=130,
        prompt_tokens=110,
    )
    llm4.input_messages = [
        make_msg("system", "You are a helpful agent."),
        make_msg("human", "Update the version in package.json."),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="read_file", args={"path": "package.json"}, id="call-1")],
        ),
        Message(role="tool", text='{"version": "1.1.0"}', tool_call_id="call-1"),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="edit_file", args={"path": "package.json"}, id="call-2")],
        ),
        Message(role="tool", text="success: true", tool_call_id="call-2"),
        Message(
            role="ai",
            text="",
            tool_calls=[ToolCall(name="read_file", args={"path": "package.json"}, id="call-3")],
        ),
        Message(role="tool", text='{"version": "1.2.0"}', tool_call_id="call-3"),
    ]
    llm4.output_message = Message(role="ai", text="Done. Version updated to 1.2.0.")

    return _make_trace("clean_execution", [root, llm1, tool1, llm2, tool2, llm3, tool3, llm4])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BUILDERS: dict[str, Callable[[], list[Run]]] = {
    "ignored_tool_error": build_ignored_tool_error,
    "unverified_completion": build_unverified_completion,
    "legitimate_repetition": build_legitimate_repetition,
    "lost_constraint_after_compaction": build_lost_constraint_after_compaction,
    "clean_execution": build_clean_execution,
}


def build_trace(trace_id: str) -> list[Run]:
    """Build a benchmark trace by ID."""
    builder = BUILDERS.get(trace_id)
    if builder is None:
        raise KeyError(f"Unknown benchmark trace: {trace_id}")
    return builder()
