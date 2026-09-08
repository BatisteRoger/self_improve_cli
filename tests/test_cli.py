"""Tests for the CLI (offline, no network)."""

import json

import pytest

from self_improve_cli.cli.main import build_parser, main
from self_improve_cli.domain import Message, RunResolution, RunType, ToolCall, Trace
from self_improve_cli.storage import TraceStore
from tests.helpers import make_msg, make_root, make_run

# ---------------------------------------------------------------------------
# Fake LangSmithSource for fetch --from-run tests
# ---------------------------------------------------------------------------


class _FakeSource:
    """Minimal fake source for offline fetch tests.

    Returns a fixed trace_id (and optional project_id) when resolving a
    run_id, and a small synthetic trace when fetching.
    """

    def __init__(self, project_name=None, **_kwargs):
        self.project_name = project_name
        self.fetch_calls: list[dict] = []

    def resolve_trace_id(self, run_id: str) -> RunResolution:
        if run_id == "run-abc":
            return RunResolution(trace_id="trace-resolved", project_id="proj-uuid-abc")
        if run_id == "run-no-proj":
            return RunResolution(trace_id="trace-no-proj", project_id=None)
        raise ValueError(f"Unknown run_id: {run_id}")

    def fetch_trace(self, trace_id: str, project_id: str | None = None) -> Trace:
        self.fetch_calls.append({"trace_id": trace_id, "project_id": project_id})
        return Trace(
            trace_id=trace_id,
            runs=[make_root(trace_id=trace_id)],
            sanitized=False,
            source="test",
        )

    def list_projects(self, limit: int = 50):
        return []


@pytest.fixture
def saved_trace(tmp_path, monkeypatch):
    """Save a sanitized trace under a temp data dir."""
    trace = Trace(
        trace_id="trace-1",
        runs=[
            make_root(trace_id="trace-1"),
            make_run(
                "llm-1",
                run_type=RunType.LLM,
                name="ChatOpenAI",
                parent="root",
                trace_id="trace-1",
                total_tokens=50,
                prompt_tokens=40,
                input_messages=[
                    make_msg("system", "You are a helpful agent."),
                    make_msg("human", "What is 2+2?"),
                ],
                output_message=Message(
                    role="ai",
                    text="",
                    tool_calls=[
                        ToolCall(name="calculator", args={"expression": "2+2"}, id="call-1")
                    ],
                ),
            ),
            make_run(
                "tool-1",
                run_type=RunType.TOOL,
                name="calculator",
                parent="root",
                trace_id="trace-1",
                inputs={"expression": "2+2"},
                outputs={"output": {"update": {"messages": [{"content": "4", "type": "tool"}]}}},
            ),
            make_run(
                "llm-2",
                run_type=RunType.LLM,
                name="ChatOpenAI",
                parent="root",
                trace_id="trace-1",
                total_tokens=50,
                prompt_tokens=45,
                input_messages=[
                    make_msg("system", "You are a helpful agent."),
                    make_msg("human", "What is 2+2?"),
                    Message(
                        role="ai",
                        text="",
                        tool_calls=[
                            ToolCall(name="calculator", args={"expression": "2+2"}, id="call-1")
                        ],
                    ),
                    Message(role="tool", text="4", tool_call_id="call-1"),
                ],
                output_message=Message(role="ai", text="2+2 = 4."),
            ),
        ],
        sanitized=True,
        sanitization_report={
            "entity_counts": {},
            "complete": True,
            "recognizer_version": "regex-fallback",
        },
        source="test",
    )
    store = TraceStore(data_root=tmp_path)
    store.save_sanitized(trace)
    return trace.trace_id, tmp_path


def test_skeleton_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["skeleton", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "[llm]" in out


def test_narrative_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["narrative", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "## Task" in out
    assert "2+2 = 4." in out


def test_run_detail_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["run-detail", trace_id, "run-llm-2", "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "You are a helpful agent." in out


def test_skeleton_json_is_structured(saved_trace, capsys):
    """skeleton --format json returns structured data, not markdown wrapped in JSON."""
    trace_id, data_dir = saved_trace
    assert main(["skeleton", trace_id, "--data-dir", str(data_dir), "--format", "json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    # Must be structured, not {"content": "<markdown>"}
    assert "content" not in data
    assert "runs" in data
    assert "trace_id" in data
    assert isinstance(data["runs"], list)
    assert len(data["runs"]) > 0
    # Each run must have an id (composable: agent can extract run IDs)
    first = data["runs"][0]
    assert "id" in first
    assert "run_type" in first
    assert "name" in first


def test_run_detail_json_is_structured(saved_trace, capsys):
    """run-detail --format json returns structured data, not markdown wrapped in JSON."""
    trace_id, data_dir = saved_trace
    assert main(
        ["run-detail", trace_id, "run-llm-2", "--data-dir", str(data_dir), "--format", "json"]
    ) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    # Must be structured, not {"content": "<markdown>"}
    assert "content" not in data
    assert data["id"] == "run-llm-2"
    assert data["run_type"] == "llm"
    # Input messages must be structured objects, not prose
    assert "input_messages" in data
    assert isinstance(data["input_messages"], list)
    assert len(data["input_messages"]) > 0
    msg = data["input_messages"][0]
    assert "role" in msg
    assert "text" in msg


# ---------------------------------------------------------------------------
# context-at command tests (SLN-6)
# ---------------------------------------------------------------------------


def test_context_at_command(saved_trace, capsys):
    """`self-improve context-at <trace_id> 0` shows the first step's input messages."""
    trace_id, data_dir = saved_trace
    assert main(["context-at", trace_id, "0", "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "Context-at step 0" in out
    assert "You are a helpful agent." in out


def test_context_at_command_step_one(saved_trace, capsys):
    """`self-improve context-at <trace_id> 1` shows the tool result at step 1."""
    trace_id, data_dir = saved_trace
    assert main(["context-at", trace_id, "1", "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "Context-at step 1" in out


def test_context_at_inputs_only(saved_trace, capsys):
    """`--inputs-only` omits the output section."""
    trace_id, data_dir = saved_trace
    assert main(
        ["context-at", trace_id, "0", "--inputs-only", "--data-dir", str(data_dir)]
    ) == 0
    out = capsys.readouterr().out
    assert "## Input messages" in out
    assert "## Output" not in out


def test_context_at_tool_filter(saved_trace, capsys):
    """`--tool <id>` isolates one tool result."""
    trace_id, data_dir = saved_trace
    assert main(
        ["context-at", trace_id, "1", "--tool", "call-1", "--data-dir", str(data_dir)]
    ) == 0
    out = capsys.readouterr().out
    assert "tool_call_id=call-1" in out
    assert "You are a helpful agent." not in out


def test_context_at_diff(saved_trace, capsys):
    """`--from 0 --to 1` shows the diff between two steps."""
    trace_id, data_dir = saved_trace
    assert main(
        [
            "context-at",
            trace_id,
            "0",
            "--from",
            "0",
            "--to",
            "1",
            "--data-dir",
            str(data_dir),
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "diff" in out.lower()


def test_context_at_json_is_structured(saved_trace, capsys):
    """`context-at --format json` returns structured data, not markdown wrapped in JSON."""
    trace_id, data_dir = saved_trace
    assert main(
        ["context-at", trace_id, "0", "--data-dir", str(data_dir), "--format", "json"]
    ) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "content" not in data
    assert data["step"] == 0
    assert data["run_id"] == "run-llm-1"
    assert isinstance(data["messages"], list)
    assert data["messages"][0]["role"] == "system"
    assert "navigation" in data


def test_context_at_out_of_range_returns_error(saved_trace, capsys):
    """An out-of-range step returns exit code 1 with a recoverable error."""
    trace_id, data_dir = saved_trace
    assert main(
        ["context-at", trace_id, "99", "--data-dir", str(data_dir)]
    ) == 1
    err = capsys.readouterr().err
    assert "out of range" in err


# ---------------------------------------------------------------------------
# target-timeline + error-neighborhood command tests (SLN-12)
# ---------------------------------------------------------------------------


def test_target_timeline_command(saved_trace, capsys):
    """`target-timeline <trace_id> <target>` finds matching runs."""
    trace_id, data_dir = saved_trace
    assert main(
        ["target-timeline", trace_id, "calculator", "--data-dir", str(data_dir)]
    ) == 0
    out = capsys.readouterr().out
    assert "Target timeline" in out
    assert "calculator" in out


def test_target_timeline_json_is_structured(saved_trace, capsys):
    """target-timeline --format json returns structured data."""
    trace_id, data_dir = saved_trace
    assert main(
        ["target-timeline", trace_id, "calculator", "--data-dir", str(data_dir), "--format", "json"]
    ) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "content" not in data
    assert data["target"] == "calculator"
    assert isinstance(data["touches"], list)


def test_error_neighborhood_command_no_errors(saved_trace, capsys):
    """`error-neighborhood` reports cleanly when there are no errors."""
    trace_id, data_dir = saved_trace
    assert main(
        ["error-neighborhood", trace_id, "--data-dir", str(data_dir)]
    ) == 0
    out = capsys.readouterr().out
    assert "No agent errors" in out


def test_error_neighborhood_json_is_structured(saved_trace, capsys):
    """error-neighborhood --format json returns structured data."""
    trace_id, data_dir = saved_trace
    assert main(
        ["error-neighborhood", trace_id, "--data-dir", str(data_dir), "--format", "json"]
    ) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "content" not in data
    assert data["trace_id"] == trace_id
    assert "errors" in data
    assert isinstance(data["errors"], list)


def test_tool_metrics_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["tool-metrics", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "# Tool Metrics" in out


def test_context_metrics_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["context-metrics", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "# Context Metrics" in out


def test_narrative_json_is_structured(saved_trace, capsys):
    """narrative --format json returns structured data, not markdown wrapped in JSON."""
    trace_id, data_dir = saved_trace
    assert main(
        ["narrative", trace_id, "--data-dir", str(data_dir), "--format", "json"]
    ) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "content" not in data
    assert data["trace_id"] == trace_id
    assert isinstance(data["steps"], list)
    assert len(data["steps"]) > 0
    assert "task" in data
    assert "mode" in data


def test_tool_metrics_json_is_structured(saved_trace, capsys):
    """tool-metrics --format json returns structured data, not markdown wrapped in JSON."""
    trace_id, data_dir = saved_trace
    assert main(
        ["tool-metrics", trace_id, "--data-dir", str(data_dir), "--format", "json"]
    ) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "content" not in data
    assert data["trace_id"] == trace_id
    assert "call_frequency" in data
    assert isinstance(data["call_frequency"], dict)


def test_context_metrics_json_is_structured(saved_trace, capsys):
    """context-metrics --format json returns structured data, not markdown wrapped in JSON."""
    trace_id, data_dir = saved_trace
    assert main(
        ["context-metrics", trace_id, "--data-dir", str(data_dir), "--format", "json"]
    ) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "content" not in data
    assert data["trace_id"] == trace_id
    assert "token_decomposition" in data
    assert isinstance(data["token_decomposition"], list)
    assert "growth_curve" in data
    assert "notes" in data


def test_skill_metrics_json_is_structured(saved_trace, capsys):
    """skill-metrics --format json returns structured data, not markdown wrapped in JSON."""
    trace_id, data_dir = saved_trace
    assert main(
        ["skill-metrics", trace_id, "--data-dir", str(data_dir), "--format", "json"]
    ) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "content" not in data
    assert data["trace_id"] == trace_id
    assert "invocations" in data


def test_info_command(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["info", trace_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["trace_id"] == trace_id
    assert data["sanitized"] is True


def test_info_command_json_format(saved_trace, capsys):
    trace_id, data_dir = saved_trace
    assert main(["info", trace_id, "--data-dir", str(data_dir), "--format", "json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["sanitized"] is True


def test_missing_trace_returns_error(saved_trace, capsys):
    _, data_dir = saved_trace
    assert main(["narrative", "unknown-trace", "--data-dir", str(data_dir)]) == 1
    err = capsys.readouterr().err
    assert "fetch" in err


def test_help_mentions_privacy():
    parser = build_parser()
    assert "anonymized" in parser.epilog.lower() or "privacy" in parser.epilog.lower()


# ---------------------------------------------------------------------------
# Skill command tests
# ---------------------------------------------------------------------------


def test_skill_list(capsys):
    """`self-improve skill` lists available skills."""
    # This works because tests run from the repo root where skills/ exists
    assert main(["skill"]) == 0
    out = capsys.readouterr().out
    assert "navigate-traces" in out
    assert "analyze-agent" in out
    assert "document-ati" in out


def test_skill_print_specific(capsys):
    """`self-improve skill navigate-traces` prints the skill content."""
    assert main(["skill", "navigate-traces"]) == 0
    out = capsys.readouterr().out
    assert "Navigate traces by question" in out
    assert "L0" in out


def test_skill_unknown(capsys):
    """`self-improve skill unknown` returns an error."""
    assert main(["skill", "nonexistent-skill"]) == 1
    err = capsys.readouterr().err
    assert "not found" in err


# ---------------------------------------------------------------------------
# Prompt command tests
# ---------------------------------------------------------------------------


def test_prompt_save_and_list(tmp_path, capsys):
    """Prompt storage: save a prompt and list it."""
    store = TraceStore(data_root=tmp_path)
    store.save_prompt("my-prompt", "prod", "You are a helpful agent.")
    store.save_prompt("my-prompt", "test", "You are a test agent.")
    store.save_prompt("other-prompt", "latest", "Another prompt.")

    prompts = store.list_prompts()
    assert len(prompts) == 3
    names = {(p["name"], p["tag"]) for p in prompts}
    assert ("my-prompt", "prod") in names
    assert ("my-prompt", "test") in names
    assert ("other-prompt", "latest") in names


def test_prompt_load(tmp_path):
    """Prompt storage: load a saved prompt."""
    store = TraceStore(data_root=tmp_path)
    store.save_prompt("my-prompt", "prod", "You are a helpful agent.")
    content = store.load_prompt("my-prompt", tag="prod")
    assert content == "You are a helpful agent."
    assert store.load_prompt("nonexistent") is None


def test_prompt_pull_disabled_by_default(capsys, monkeypatch):
    """`prompt pull` is blocked when ENABLE_PROMPT_HUB is not true."""
    monkeypatch.delenv("ENABLE_PROMPT_HUB", raising=False)
    # Prevent _load_env from reloading .env and re-setting the var
    monkeypatch.setattr("self_improve_cli.cli.main._load_env", lambda: None)
    assert main(["prompt", "pull", "test-prompt"]) == 1
    err = capsys.readouterr().err
    assert "disabled" in err.lower()


def test_prompt_pull_disabled_when_false(capsys, monkeypatch):
    """`prompt pull` is blocked when ENABLE_PROMPT_HUB=false."""
    monkeypatch.setenv("ENABLE_PROMPT_HUB", "false")
    monkeypatch.setattr("self_improve_cli.cli.main._load_env", lambda: None)
    assert main(["prompt", "pull", "test-prompt"]) == 1
    err = capsys.readouterr().err
    assert "disabled" in err.lower()


def test_prompt_show_command(tmp_path, capsys):
    """`self-improve prompt show` prints a saved prompt."""
    store = TraceStore(data_root=tmp_path)
    store.save_prompt("my-prompt", "latest", "You are a helpful agent.")
    assert main(["prompt", "show", "my-prompt", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "helpful agent" in out


def test_prompt_show_not_found(tmp_path, capsys):
    """`self-improve prompt show` errors for unknown prompts."""
    assert main(["prompt", "show", "nonexistent", "--data-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_prompt_list_command(tmp_path, capsys):
    """`self-improve prompt list` lists saved prompts."""
    store = TraceStore(data_root=tmp_path)
    store.save_prompt("my-prompt", "prod", "You are a helpful agent.")
    assert main(["prompt", "list", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "my-prompt:prod" in out


def test_prompt_list_empty(tmp_path, capsys):
    """`self-improve prompt list` handles no saved prompts."""
    assert main(["prompt", "list", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "No saved prompts" in out


def test_prompt_diff_command(saved_trace, tmp_path, capsys):
    """`self-improve prompt diff` compares a saved prompt against a trace."""
    trace_id, data_dir = saved_trace
    # Save a prompt that differs from the trace's system message
    store = TraceStore(data_root=tmp_path)
    store.save_prompt("my-prompt", "latest", "You are a different agent.")
    assert main(["prompt", "diff", "my-prompt", trace_id, "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Approximate diff" in out


def test_prompt_diff_identical(saved_trace, tmp_path, capsys):
    """`self-improve prompt diff` reports no differences when identical."""
    trace_id, _ = saved_trace
    # Save a prompt that matches the trace's system message
    store = TraceStore(data_root=tmp_path)
    store.save_prompt("my-prompt", "latest", "You are a helpful agent.")
    assert main(["prompt", "diff", "my-prompt", trace_id, "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "No differences" in out


# ---------------------------------------------------------------------------
# ATI command tests
# ---------------------------------------------------------------------------


def test_ati_save_and_list(tmp_path):
    """ATI storage: list registered ATIs."""
    store = TraceStore(data_root=tmp_path)
    # Create an ATI document manually (as the document-ati skill would)
    ati_dir = tmp_path / "ati" / "my-agent"
    ati_dir.mkdir(parents=True)
    (ati_dir / "architecture.md").write_text("# My Agent\n\nPurpose: test.", encoding="utf-8")

    atis = store.list_atis()
    assert atis == ["my-agent"]


def test_ati_load(tmp_path):
    """ATI storage: load an architecture document."""
    store = TraceStore(data_root=tmp_path)
    ati_dir = tmp_path / "ati" / "my-agent"
    ati_dir.mkdir(parents=True)
    (ati_dir / "architecture.md").write_text("# My Agent\n\nPurpose: test.", encoding="utf-8")

    content = store.load_ati("my-agent")
    assert content is not None
    assert "My Agent" in content
    assert store.load_ati("nonexistent") is None


def test_ati_list_command(tmp_path, capsys):
    """`self-improve ati list` lists registered ATIs."""
    ati_dir = tmp_path / "ati" / "my-agent"
    ati_dir.mkdir(parents=True)
    (ati_dir / "architecture.md").write_text("# My Agent", encoding="utf-8")

    assert main(["ati", "list", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "my-agent" in out


def test_ati_list_empty(tmp_path, capsys):
    """`self-improve ati list` handles no ATIs."""
    assert main(["ati", "list", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "No ATIs" in out


def test_ati_show_command(tmp_path, capsys):
    """`self-improve ati show` prints an architecture document."""
    ati_dir = tmp_path / "ati" / "my-agent"
    ati_dir.mkdir(parents=True)
    (ati_dir / "architecture.md").write_text("# My Agent\n\nPurpose: test.", encoding="utf-8")

    assert main(["ati", "show", "my-agent", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "My Agent" in out


def test_ati_show_not_found(tmp_path, capsys):
    """`self-improve ati show` errors for unknown ATIs."""
    assert main(["ati", "show", "nonexistent", "--data-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "not found" in err


# ---------------------------------------------------------------------------
# fetch --from-run tests
# ---------------------------------------------------------------------------


def test_fetch_from_run_resolves_and_fetches(tmp_path, capsys, monkeypatch):
    """`fetch --from-run <run_id>` resolves the run_id to a trace_id and fetches."""
    import self_improve_cli.sources.langsmith as ls_module

    monkeypatch.setattr(ls_module, "LangSmithSource", _FakeSource)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    rc = main(
        [
            "fetch",
            "run-abc",
            "--from-run",
            "--data-dir",
            str(tmp_path),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["trace_id"] == "trace-resolved"
    assert data["resolved_from_run"] == "run-abc"
    assert data["runs"] == 1


def test_fetch_from_run_resolution_message_on_stderr(tmp_path, capsys, monkeypatch):
    """`fetch --from-run` prints the resolution to stderr (diagnostics channel)."""
    import self_improve_cli.sources.langsmith as ls_module

    monkeypatch.setattr(ls_module, "LangSmithSource", _FakeSource)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    rc = main(
        [
            "fetch",
            "run-abc",
            "--from-run",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "Resolved run run-abc -> trace trace-resolved" in err


def test_fetch_without_from_run_does_not_resolve(tmp_path, capsys, monkeypatch):
    """`fetch <trace_id>` (without --from-run) does not call resolve_trace_id."""
    import self_improve_cli.sources.langsmith as ls_module

    monkeypatch.setattr(ls_module, "LangSmithSource", _FakeSource)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    rc = main(
        [
            "fetch",
            "trace-resolved",
            "--data-dir",
            str(tmp_path),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["trace_id"] == "trace-resolved"
    assert "resolved_from_run" not in data


def test_fetch_from_run_passes_project_id_to_fetch(tmp_path, capsys, monkeypatch):
    """`fetch --from-run` passes the resolved project_id to fetch_trace."""
    import self_improve_cli.sources.langsmith as ls_module

    fake_instances: list[_FakeSource] = []

    class _CapturingFakeSource(_FakeSource):
        def __init__(self, project_name=None, **_kwargs):
            super().__init__(project_name, **_kwargs)
            fake_instances.append(self)

    monkeypatch.setattr(ls_module, "LangSmithSource", _CapturingFakeSource)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    rc = main(
        [
            "fetch",
            "run-abc",
            "--from-run",
            "--data-dir",
            str(tmp_path),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert len(fake_instances) == 1
    assert fake_instances[0].fetch_calls == [
        {"trace_id": "trace-resolved", "project_id": "proj-uuid-abc"}
    ]


def test_fetch_from_run_without_project_id(tmp_path, capsys, monkeypatch):
    """`fetch --from-run` works when the resolved run has no session_id."""
    import self_improve_cli.sources.langsmith as ls_module

    monkeypatch.setattr(ls_module, "LangSmithSource", _FakeSource)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    rc = main(
        [
            "fetch",
            "run-no-proj",
            "--from-run",
            "--data-dir",
            str(tmp_path),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["trace_id"] == "trace-no-proj"


def test_fetch_empty_trace_warns_on_stderr(tmp_path, capsys, monkeypatch):
    """`fetch` warns on stderr when the trace has 0 runs."""
    import self_improve_cli.sources.langsmith as ls_module

    class _EmptySource(_FakeSource):
        def fetch_trace(self, trace_id: str, project_id: str | None = None) -> Trace:
            self.fetch_calls.append({"trace_id": trace_id, "project_id": project_id})
            return Trace(trace_id=trace_id, runs=[], sanitized=False, source="test")

    monkeypatch.setattr(ls_module, "LangSmithSource", _EmptySource)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    rc = main(
        [
            "fetch",
            "trace-empty",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "No runs found" in err
    assert "list-projects" in err


def test_fetch_from_run_empty_trace_warns_with_project_id(tmp_path, capsys, monkeypatch):
    """`fetch --from-run` warns with project_id context when 0 runs are found."""
    import self_improve_cli.sources.langsmith as ls_module

    class _EmptySource(_FakeSource):
        def fetch_trace(self, trace_id: str, project_id: str | None = None) -> Trace:
            self.fetch_calls.append({"trace_id": trace_id, "project_id": project_id})
            return Trace(trace_id=trace_id, runs=[], sanitized=False, source="test")

    monkeypatch.setattr(ls_module, "LangSmithSource", _EmptySource)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    rc = main(
        [
            "fetch",
            "run-abc",
            "--from-run",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "No runs found" in err
    assert "proj-uuid-abc" in err


# ---------------------------------------------------------------------------
# list-projects command tests
# ---------------------------------------------------------------------------


def test_list_projects_command(tmp_path, capsys, monkeypatch):
    """`list-projects` lists accessible LangSmith projects."""
    import self_improve_cli.sources.langsmith as ls_module

    class _ProjectsSource(_FakeSource):
        def list_projects(self, limit: int = 50):
            return [
                type("P", (), {"id": "uuid-1", "name": "my-agent-dev", "run_count": 42})(),
                type("P", (), {"id": "uuid-2", "name": "my-agent-prod", "run_count": None})(),
            ]

    monkeypatch.setattr(ls_module, "LangSmithSource", _ProjectsSource)

    rc = main(["list-projects", "--data-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "my-agent-dev" in out
    assert "uuid-1" in out
    assert "runs=42" in out
    assert "my-agent-prod" in out


def test_list_projects_json_format(tmp_path, capsys, monkeypatch):
    """`list-projects --format json` outputs structured JSON."""
    import self_improve_cli.sources.langsmith as ls_module

    class _ProjectsSource(_FakeSource):
        def list_projects(self, limit: int = 50):
            return [
                type("P", (), {"id": "uuid-1", "name": "my-agent-dev", "run_count": 42})(),
            ]

    monkeypatch.setattr(ls_module, "LangSmithSource", _ProjectsSource)

    rc = main(["list-projects", "--data-dir", str(tmp_path), "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data == [{"id": "uuid-1", "name": "my-agent-dev", "run_count": 42}]
