"""`thread` command: URL parsing, adapter listing, anonymized CLI output (synthetic, no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest

import self_improve_cli.sources.langsmith as ls_module
from self_improve_cli.cli.main import main
from self_improve_cli.sources.langsmith import LangSmithSource
from self_improve_cli.sources.langsmith_url import parse_langsmith_thread_url

ORG = "11111111-1111-4111-8111-111111111111"
PROJ = "22222222-2222-4222-8222-222222222222"
THREAD = "55555555-5555-4555-8555-555555555555"
URL = (
    f"https://smith.example.com/o/{ORG}/a/some-project/projects/p/{PROJ}"
    f"?peek=66666666-6666-4666-8666-666666666666&peekedConversationId={THREAD}"
)


class _Pages:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item

        return gen()


def _item(trace_id, minute, *, error=None, ended=True, user="hello", answer="ai: hi"):
    start = datetime(2026, 1, 2, 10, minute, 0, tzinfo=UTC)
    end = datetime(2026, 1, 2, 10, minute, 4, 500000, tzinfo=UTC) if ended else None
    return NS(
        trace_id=trace_id,
        name="agent",
        start_time=start,
        end_time=end,
        total_tokens=100 + minute,
        error=error,
        inputs_preview=user,
        outputs_preview=answer,
    )


def _client(items, calls):
    def list_traces(thread_id, **kw):
        calls.append((thread_id, kw))
        return _Pages(items)

    async def aread_project(project_name=None, project_id=None):
        calls.append(("aread_project", project_name, project_id))
        return NS(id=project_id or PROJ, name=project_name or "demo-project")

    return NS(threads=NS(list_traces=list_traces), aread_project=aread_project)


# --- URL parsing ---------------------------------------------------------


def test_parse_thread_url_extracts_thread_and_project():
    ref = parse_langsmith_thread_url(URL)
    assert (ref.thread_id, ref.project_id) == (THREAD, PROJ)


def test_parse_thread_url_keeps_non_uuid_thread_id():
    ref = parse_langsmith_thread_url("https://h/x?peekedConversationId=conv-42")
    assert (ref.thread_id, ref.project_id) == ("conv-42", None)


def test_parse_thread_url_without_thread_id_raises():
    with pytest.raises(ValueError, match="peekedConversationId"):
        parse_langsmith_thread_url(f"https://h/o/{ORG}/projects/p/{PROJ}?peek=x")


# --- adapter -------------------------------------------------------------


def test_list_thread_turns_orders_and_derives_status(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    calls: list = []
    items = [
        _item("t3", 5, ended=False),
        _item("t1", 1),
        _item("t2", 3, error="boom"),
    ]
    src = LangSmithSource(client=_client(items, calls))
    listing = src.list_thread_turns(THREAD, project_id=PROJ)
    assert [r.trace_id for r in listing.runs] == ["t1", "t2", "t3"]
    assert [r.status for r in listing.runs] == ["success", "error", "pending"]
    assert listing.project_name == "demo-project"
    assert not listing.truncated
    thread_call = next(c for c in calls if c[0] == THREAD)
    assert thread_call[1]["project_id"] == PROJ
    # Previews only: full inputs/outputs never cross the boundary.
    assert "INPUTS" not in thread_call[1]["selects"]
    assert "OUTPUTS" not in thread_call[1]["selects"]


def test_list_thread_turns_truncates_at_limit(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    items = [_item(f"t{i}", i) for i in range(5)]
    src = LangSmithSource(client=_client(items, []))
    listing = src.list_thread_turns(THREAD, project_id=PROJ, limit=3)
    assert len(listing.runs) == 3 and listing.truncated


def test_list_thread_turns_applies_allowlist_to_url_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "other-*")
    src = LangSmithSource(client=_client([], []))
    with pytest.raises(PermissionError):
        src.list_thread_turns(THREAD, project_id=PROJ)


def test_list_thread_turns_without_project_raises(monkeypatch):
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    src = LangSmithSource(client=_client([], []))
    with pytest.raises(RuntimeError, match="--project"):
        src.list_thread_turns(THREAD)


# --- CLI -----------------------------------------------------------------


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    monkeypatch.delenv("SELF_IMPROVE_DROP_KEYS", raising=False)
    monkeypatch.delenv("SELF_IMPROVE_VALUE_MAP", raising=False)
    items = [
        _item("t1", 1, user="my mail is jane.doe@example.com"),
        _item("t2", 2, user="again jane.doe@example.com", answer="ai: done"),
    ]
    calls: list = []
    monkeypatch.setattr(ls_module, "_get_client", lambda *a, **kw: _client(items, calls))
    return calls


def test_thread_cli_json_anonymizes_excerpts(fake_client, capsys):
    rc = main(["thread", URL, "--format", "json", "--anonymizer", "regex"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["thread_id"] == THREAD and out["project_id"] == PROJ
    assert [t["trace_id"] for t in out["turns"]] == ["t1", "t2"]
    first, second = out["turns"]
    assert first["latency_s"] == 4.5 and first["total_tokens"] == 101
    assert "jane.doe@example.com" not in json.dumps(out)
    # One pass over the thread: the same value gets the same placeholder in every turn.
    placeholder = first["input_excerpt"].removeprefix("my mail is ")
    assert placeholder in second["input_excerpt"]
    assert out["sanitization_report"]["entity_counts"]["EMAIL"] == 2
    assert out["next"] == ["self-improve fetch <trace_id> --project 'demo-project'"]


def test_thread_cli_markdown_lists_one_line_per_turn(fake_client, capsys):
    rc = main(["thread", THREAD, "--project", "demo-project", "--anonymizer", "regex"])
    assert rc == 0
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("| 1 ") or line.startswith("| 2 ")]
    assert len(rows) == 2 and "| t1 |" in rows[0] and "ai: done" in rows[1]
    assert "jane.doe@example.com" not in out


def test_thread_cli_empty_thread_is_recoverable_error(monkeypatch, capsys):
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    monkeypatch.setattr(ls_module, "_get_client", lambda *a, **kw: _client([], []))
    rc = main(["thread", URL, "--anonymizer", "regex"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no traces found" in err and "--project" in err
