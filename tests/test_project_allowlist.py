"""Tests for the LangSmith project allowlist."""

import pytest

from self_improve_cli.sources.langsmith import (
    LangSmithSource,
    _check_project_allowed,
    _get_allowed_projects,
)


def test_no_allowlist_allows_all(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    assert _get_allowed_projects() == []
    # Should not raise
    _check_project_allowed("anything")
    _check_project_allowed("production")


def test_empty_allowlist_allows_all(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "")
    assert _get_allowed_projects() == []
    _check_project_allowed("anything")


def test_exact_match_allowed(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "my-agent-dev,my-agent-test")
    _check_project_allowed("my-agent-dev")
    _check_project_allowed("my-agent-test")


def test_glob_pattern_allowed(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "staging*")
    _check_project_allowed("staging")
    _check_project_allowed("staging-my-app")
    _check_project_allowed("staging.v2")


def test_non_matching_project_rejected(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "staging*")
    with pytest.raises(PermissionError, match="not in the allowed list"):
        _check_project_allowed("production")


def test_production_blocked_when_only_staging_allowed(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "staging*,my-agent-dev")
    with pytest.raises(PermissionError, match="production"):
        _check_project_allowed("production")


def test_question_mark_pattern(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "my-agent-?")
    _check_project_allowed("my-agent-a")
    _check_project_allowed("my-agent-x")
    with pytest.raises(PermissionError):
        _check_project_allowed("my-agent-dev")  # too long for ?


def test_whitespace_in_patterns_handled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", " my-agent-dev , my-agent-test , ")
    assert _get_allowed_projects() == ["my-agent-dev", "my-agent-test"]
    _check_project_allowed("my-agent-dev")


def test_resolve_project_enforces_allowlist(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "staging*")
    source = LangSmithSource(project_name="production")
    with pytest.raises(PermissionError, match="production"):
        source._resolve_project()


def test_resolve_project_passes_when_allowed(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "staging*")
    source = LangSmithSource(project_name="staging-app")
    assert source._resolve_project() == "staging-app"


def test_resolve_project_passes_when_no_allowlist(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    source = LangSmithSource(project_name="anything")
    assert source._resolve_project() == "anything"
