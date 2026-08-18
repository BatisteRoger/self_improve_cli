"""Tests for build_tools_overview and build_tools_detail representations."""

from __future__ import annotations

from self_improve_cli.domain import RunType
from self_improve_cli.representations import build_tools_detail, build_tools_overview
from tests.helpers import make_llm, make_root, make_run


def _make_tool_def(name: str, description: str = "", params: dict | None = None) -> dict:
    """Build a minimal tool definition matching the LangSmith format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


def _llm_with_tools(idx: str, tools: list[dict], name: str = "ChatOpenAI") -> object:
    """Build an LLM run with tools in extra.extra.invocation_params.tools."""
    return make_run(
        idx,
        run_type=RunType.LLM,
        name=name,
        extra={"extra": {"invocation_params": {"tools": tools}}},
    )


class TestBuildToolsOverview:
    def test_empty_trace(self):
        result = build_tools_overview([])
        assert "no LLM runs" in result

    def test_no_tools_runs(self):
        runs = [make_root(), make_llm("1", 100)]
        result = build_tools_overview(runs)
        assert "guardrails" in result
        assert "0 unique tool" in result or "1 scope" in result

    def test_two_scopes_with_matrix(self):
        supervisor_tools = [
            _make_tool_def("search_agent", "desc-a"),
            _make_tool_def("financial_agent", "desc-b"),
        ]
        search_tools = [
            _make_tool_def("rag_tool", "desc-c"),
            _make_tool_def("web_search_tool_safe", "desc-d"),
        ]
        runs = [
            make_root(),
            _llm_with_tools("1", supervisor_tools),
            _llm_with_tools("2", search_tools),
        ]
        result = build_tools_overview(runs)
        assert "supervisor" in result
        assert "search_agent" in result
        assert "rag_tool" in result
        assert "search_agent" in result  # tool name appears too
        # Matrix marks
        assert "X" in result

    def test_groups_by_tool_set_signature(self):
        tools = [_make_tool_def("rag_tool", "desc")]
        runs = [
            make_root(),
            _llm_with_tools("1", tools),
            _llm_with_tools("2", tools),  # same set → same group
        ]
        result = build_tools_overview(runs)
        assert "2 run" in result  # "2 runs" in the scope label


class TestBuildToolsDetail:
    def test_empty_trace(self):
        result = build_tools_detail([])
        assert "no LLM runs" in result

    def test_specific_tool_not_found(self):
        runs = [make_root(), make_llm("1", 100)]
        result = build_tools_detail(runs, tool_name="nonexistent")
        assert "not found" in result

    def test_specific_tool_found_with_docstring(self):
        tools = [
            _make_tool_def(
                "rag_tool",
                "A RAG tool for documents.",
                {
                    "type": "object",
                    "properties": {"question": {"type": "string", "description": "The question"}},
                    "required": ["question"],
                },
            ),
        ]
        runs = [make_root(), _llm_with_tools("1", tools)]
        result = build_tools_detail(runs, tool_name="rag_tool")
        assert "## rag_tool" in result
        assert "A RAG tool for documents." in result
        assert "**question**" in result
        assert "(required)" in result

    def test_all_tools_without_name(self):
        tools = [
            _make_tool_def("tool_a", "Description A"),
            _make_tool_def("tool_b", "Description B"),
        ]
        runs = [make_root(), _llm_with_tools("1", tools)]
        result = build_tools_detail(runs)
        assert "## tool_a" in result
        assert "## tool_b" in result
        assert "Description A" in result
        assert "Description B" in result

    def test_scope_labels_in_detail(self):
        supervisor_tools = [_make_tool_def("search_agent", "desc")]
        # Use both rag_tool + web_search_tool_safe to match the search_agent pattern
        search_tools = [
            _make_tool_def("rag_tool", "desc"),
            _make_tool_def("web_search_tool_safe", "desc"),
        ]
        runs = [
            make_root(),
            _llm_with_tools("1", supervisor_tools),
            _llm_with_tools("2", search_tools),
        ]
        result = build_tools_detail(runs, tool_name="rag_tool")
        assert "search_agent" in result  # scope label
