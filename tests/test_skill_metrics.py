"""Tests for skill metrics."""

from self_improve_cli.metrics.skill_metrics import (
    build_skill_metrics,
    extract_skill_name,
    skill_invocations,
    skill_token_cost,
)
from tests.helpers import make_llm, make_root, make_tool

# --- extract_skill_name -----------------------------------------------------


def test_extract_skill_name_from_path():
    assert extract_skill_name("/priorisation-financiere/SKILL.md") == "priorisation-financiere"


def test_extract_skill_name_nested_path():
    assert (
        extract_skill_name("/skills/priorisation-financiere/SKILL.md") == "priorisation-financiere"
    )


def test_extract_skill_name_backslash_path():
    assert extract_skill_name("\\skills\\priorisation-financiere\\SKILL.md") == (
        "priorisation-financiere"
    )


def test_extract_skill_name_non_skill_path():
    assert extract_skill_name("/some/file.md") is None


def test_extract_skill_name_empty():
    assert extract_skill_name("") is None


def test_extract_skill_name_skill_md_only():
    # Just "SKILL.md" with no parent directory
    assert extract_skill_name("SKILL.md") is None


# --- skill_invocations ------------------------------------------------------


def test_skill_invocations_detects_read_file():
    runs = [
        make_root(),
        make_tool("1", "read_file", {"path": "/priorisation-financiere/SKILL.md"}),
    ]
    result = skill_invocations(runs)
    assert len(result) == 1
    assert result[0]["skill_name"] == "priorisation-financiere"
    assert result[0]["tool_name"] == "read_file"
    assert result[0]["file_path"] == "/priorisation-financiere/SKILL.md"


def test_skill_invocations_detects_grep():
    runs = [
        make_root(),
        make_tool("1", "grep", {"file_path": "/boost-engine/SKILL.md", "pattern": "bias"}),
    ]
    result = skill_invocations(runs)
    assert len(result) == 1
    assert result[0]["skill_name"] == "boost-engine"


def test_skill_invocations_ignores_non_skill_read():
    runs = [
        make_root(),
        make_tool("1", "read_file", {"path": "/some/other.md"}),
    ]
    assert skill_invocations(runs) == []


def test_skill_invocations_ignores_non_skill_tool_name():
    runs = [
        make_root(),
        make_tool("1", "edit_file", {"path": "/priorisation-financiere/SKILL.md"}),
    ]
    assert skill_invocations(runs) == []


def test_skill_invocations_empty_trace():
    assert skill_invocations([]) == []


def test_skill_invocations_multiple_skills():
    runs = [
        make_root(),
        make_tool("1", "read_file", {"path": "/priorisation-financiere/SKILL.md"}),
        make_tool("2", "read_file", {"path": "/reaction-baisse-marche/SKILL.md"}),
    ]
    result = skill_invocations(runs)
    assert len(result) == 2
    assert result[0]["skill_name"] == "priorisation-financiere"
    assert result[1]["skill_name"] == "reaction-baisse-marche"


def test_skill_invocations_nested_json_string_args():
    """Handle args stored as a JSON string under 'input' key (LangSmith format)."""
    import json

    nested = json.dumps({"file_path": "/priorisation-financiere/SKILL.md", "offset": 0})
    runs = [
        make_root(),
        make_tool("1", "read_file", {"input": nested}),
    ]
    result = skill_invocations(runs)
    assert len(result) == 1
    assert result[0]["skill_name"] == "priorisation-financiere"
    assert result[0]["file_path"] == "/priorisation-financiere/SKILL.md"


# --- skill_token_cost -------------------------------------------------------


def test_skill_token_cost_with_two_llm_steps():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=3500),
        make_tool("2", "read_file", {"path": "/priorisation-financiere/SKILL.md"}),
        make_llm("3", prompt_tokens=12000),
    ]
    result = skill_token_cost(runs)
    assert len(result) == 1
    assert result[0]["skill_name"] == "priorisation-financiere"
    assert result[0]["token_delta"] == 8500
    assert result[0]["extra_llm_call"] is True


def test_skill_token_cost_no_skill():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=3500),
    ]
    assert skill_token_cost(runs) == []


def test_skill_token_cost_empty_trace():
    assert skill_token_cost([]) == []


def test_skill_token_cost_no_prior_llm():
    """Skill called at the start of the trace — no previous LLM step."""
    runs = [
        make_root(),
        make_tool("1", "read_file", {"path": "/priorisation-financiere/SKILL.md"}),
        make_llm("2", prompt_tokens=12000),
    ]
    result = skill_token_cost(runs)
    assert len(result) == 1
    assert result[0]["token_delta"] is None  # can't compute delta without prior LLM


# --- build_skill_metrics ----------------------------------------------------


def test_build_skill_metrics_empty_trace():
    assert "(empty trace)" in build_skill_metrics([])


def test_build_skill_metrics_no_skills():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=3500),
    ]
    result = build_skill_metrics(runs)
    assert "No skill invocations" in result


def test_build_skill_metrics_with_skill():
    runs = [
        make_root(),
        make_llm("1", prompt_tokens=3500),
        make_tool("2", "read_file", {"path": "/priorisation-financiere/SKILL.md"}),
        make_llm("3", prompt_tokens=12000),
    ]
    result = build_skill_metrics(runs)
    assert "priorisation-financiere" in result
    assert "Invocations" in result
    assert "Token cost" in result
