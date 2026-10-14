"""Role normalization — LangChain class names must not leak into roles.

Regression: streamed LiLa runs serialize messages as AIMessageChunk, which
the source adapter lowercased to "aimessagechunk". Downstream code compares
against "ai", so tool_call_id -> tool-name resolution and AI-token
classification silently broke on real traces.
"""

from self_improve_cli.domain import Message, ToolCall
from self_improve_cli.metrics.context_metrics import token_decomposition
from self_improve_cli.sources import _normalize_message
from tests.helpers import make_llm, make_msg, make_root


def test_normalize_message_strips_messagechunk_suffix():
    lc_msg = {
        "id": ["langchain", "schema", "messages", "AIMessageChunk"],
        "kwargs": {
            "content": "hi",
            "tool_calls": [{"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}],
        },
    }
    assert _normalize_message(lc_msg)["role"] == "ai"


def test_normalize_message_still_handles_plain_message_suffix():
    lc_msg = {
        "id": ["langchain", "schema", "messages", "HumanMessage"],
        "kwargs": {"content": "hello"},
    }
    assert _normalize_message(lc_msg)["role"] == "human"


def test_message_post_init_normalizes_stored_chunk_roles():
    # Traces stored before the source fix still carry the leaked class name;
    # the canonical model normalizes at construction.
    assert Message(role="aimessagechunk").role == "ai"
    assert Message(role="HumanMessageChunk").role == "human"
    assert Message(role="system").role == "system"


def test_tool_name_resolution_with_chunk_role():
    """tool_call_id -> name resolution works on stored chunk-role messages."""
    ai_chunk = Message(
        role="aimessagechunk",
        text="",
        tool_calls=[ToolCall(name="read_file", args={"path": "a.py"}, id="call-1")],
    )
    tool_msg = Message(role="tool", text="x" * 400, tool_call_id="call-1")
    runs = [
        make_root(),
        make_llm(
            "1",
            prompt_tokens=200,
            input_messages=[make_msg("system", "sys"), ai_chunk, tool_msg],
        ),
    ]
    decomp = token_decomposition(runs)[0]
    assert decomp["per_tool"].get("read_file") == 100  # 400 chars / 4
    assert decomp["categories"]["ai"] > 0
