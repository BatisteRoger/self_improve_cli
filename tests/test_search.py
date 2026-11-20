"""Tests for `search` (SLN-69): filter building, matching, excerpts, CLI paths.

The LangSmith client is a fake; all run content is synthetic.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from self_improve_cli.cli import main as cli_main
from self_improve_cli.cli.search import anonymized_excerpt, build_matcher, output_text
from self_improve_cli.domain import Message, Run, RunType, ToolCall
from self_improve_cli.sources import langsmith as ls
from self_improve_cli.sources.langsmith import build_search_filter

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSdkRun:
    def __init__(self, **kw: Any) -> None:
        self._d = kw

    def dict(self) -> dict[str, Any]:
        return dict(self._d)


def _llm_run(run_id: str, trace_id: str, text: str, model: str | None = "model-x") -> _FakeSdkRun:
    return _FakeSdkRun(
        id=run_id,
        trace_id=trace_id,
        name="ChatModel",
        run_type="LLM",
        start_time="2026-01-02T03:04:05+00:00",
        outputs={"generations": [[{"message": {"type": "ai", "content": text}}]]},
        metadata={"ls_model_name": model} if model else {},
    )


class _Pager:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> _Pager:
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _FakeRuns:
    def __init__(self, runs: list[Any]) -> None:
        self._runs = runs
        self.calls: list[dict[str, Any]] = []

    def query(self, **kw: Any) -> _Pager:
        self.calls.append(kw)
        return _Pager(self._runs)


class _FakeProject:
    id = "proj-uuid"


class _FakeClient:
    def __init__(self, runs: list[Any]) -> None:
        self.runs = _FakeRuns(runs)

    async def aread_project(self, project_name: str | None = None, **kw: Any) -> _FakeProject:
        return _FakeProject()


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # main() loads ./.env: keep a developer .env out of tests

    def install(runs: list[Any]) -> _FakeClient:
        client = _FakeClient(runs)
        monkeypatch.setattr(ls, "_get_client", lambda workspace_id=None: client)
        monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
        monkeypatch.setenv("SELF_IMPROVE_ANONYMIZER", "regex")
        monkeypatch.delenv("SELF_IMPROVE_DROP_KEYS", raising=False)
        monkeypatch.delenv("SELF_IMPROVE_VALUE_MAP", raising=False)
        return client

    return install


def _run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli_main.main(["search", "--project", "demo", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_build_search_filter_combines_text_and_metadata() -> None:
    expr = build_search_filter('say "hi"', [("node", "model"), ("env", "test")])
    assert expr == (
        'and(search("say \\"hi\\""), '
        'and(eq(metadata_key, "node"), eq(metadata_value, "model")), '
        'and(eq(metadata_key, "env"), eq(metadata_value, "test")))'
    )
    assert build_search_filter(None, [("k", "v")]) == (
        'and(eq(metadata_key, "k"), eq(metadata_value, "v"))'
    )
    assert build_search_filter(None, []) is None


def test_output_text_includes_tool_call_args_and_falls_back_to_outputs() -> None:
    run = Run(
        id="r",
        trace_id="t",
        run_type=RunType.LLM,
        name="m",
        output_message=Message(
            role="ai", text="hello", tool_calls=[ToolCall(name="lookup", args={"q": "blue"})]
        ),
    )
    assert output_text(run) == 'hello\n{"lookup": {"q": "blue"}}'
    tool = Run(id="r", trace_id="t", run_type=RunType.TOOL, name="x", outputs={"out": "ok"})
    assert output_text(tool) == '{"out": "ok"}'


def test_matcher_text_is_case_insensitive_and_both_criteria_must_match() -> None:
    m = build_matcher("need FLAG", None)
    assert m("we Need flag here") == (3, 12)
    assert m("nothing") is None
    both = build_matcher("total", re.compile(r"\d+ ?\?"))
    assert both("total is 42 ?") == (9, 13)
    assert both("total is 42") is None


def test_excerpt_is_anonymized_and_bounded() -> None:
    from self_improve_cli.domain import Trace
    from self_improve_cli.privacy import anonymize_trace

    text = "x" * 300 + " contact jane.doe@example.com about MARKER now " + "y" * 300
    run = Run(id="r", trace_id="t", run_type=RunType.LLM, name="m")
    matcher = build_matcher("MARKER", None)
    span = matcher(text)
    assert span is not None

    def anonymize(r: Run) -> Run:
        return anonymize_trace(Trace(trace_id="t", runs=[r]), backend="regex")[0].runs[0]

    excerpt = anonymized_excerpt(run, text, span, matcher, anonymize)
    assert "jane.doe@example.com" not in excerpt
    assert "<EMAIL_" in excerpt
    assert "MARKER" in excerpt
    assert excerpt.startswith("...") and excerpt.endswith("...")
    assert len(excerpt) < 2 * 80 + len("MARKER") + 20


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_search_counts_scanned_and_matched_json(fake_client, capsys) -> None:
    client = fake_client(
        [
            _llm_run("r1", "t1", "all good"),
            _llm_run("r2", "t2", "Total is 12\u202f? write to a.b@example.org"),
            _llm_run("r3", "t3", "Total is 7\u202f?", model=None),
        ]
    )
    code, out, err = _run_cli(
        capsys,
        "--regex",
        "\\d\u202f\\?",
        "--metadata",
        "node=model",
        "--since",
        "3",
        "--limit",
        "1",
        "--format",
        "json",
    )
    assert code == 0
    data = json.loads(out)
    assert (data["scanned"], data["matched"], data["shown"], data["complete"]) == (3, 2, 1, True)
    hit = data["matches"][0]
    assert (hit["trace_id"], hit["run_id"], hit["model"]) == ("t2", "r2", "model-x")
    assert "a.b@example.org" not in hit["excerpt"]
    call = client.runs.calls[0]
    assert call["run_type"] == "LLM"
    assert call["filter"] == 'and(eq(metadata_key, "node"), eq(metadata_value, "model"))'
    assert "OUTPUTS" in call["selects"] and "INPUTS" not in call["selects"]


def test_search_markdown_reports_frequency_and_next_step(fake_client, capsys) -> None:
    fake_client([_llm_run("r1", "t1", "the answer"), _llm_run("r2", "t2", "Need flag here")])
    code, out, err = _run_cli(capsys, "--text", "need flag")
    assert code == 0
    assert "matched 1 of 2 runs scanned (50.00%)" in out
    assert "trace_id=t2  run_id=r2  model=model-x" in out
    assert "search(" in out  # the server filter is stated: scanned is not every run
    assert "next: self-improve fetch t2" in out


def test_search_stops_at_max_scan_and_says_so(fake_client, capsys) -> None:
    fake_client([_llm_run(f"r{i}", f"t{i}", "hit") for i in range(5)])
    code, out, err = _run_cli(capsys, "--text", "hit", "--max-scan", "2", "--format", "json")
    data = json.loads(out)
    assert code == 0
    assert (data["scanned"], data["complete"]) == (2, False)
    assert "--max-scan" in err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--regex", "("],
        ["--text", "x", "--metadata", "novalue"],
        ["--text", "x", "--since", "0"],
    ],
)
def test_search_usage_errors_exit_2_before_any_call(fake_client, capsys, argv) -> None:
    client = fake_client([])
    code, _, err = _run_cli(capsys, *argv)
    assert code == 2
    assert err.startswith("Error:")
    assert client.runs.calls == []
