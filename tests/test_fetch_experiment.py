"""fetch_experiment adapter: pagination, dedup, privacy boundary (synthetic, no network)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from self_improve_cli.sources.langsmith import LangSmithSource


class _Pages:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item

        return gen()


def _run(rid, score, comment="secret comment"):
    stats = {"k": NS(avg=score, comments=[comment])}
    return NS(id=rid, trace_id=rid, status="success", feedback_stats=stats)


def _client(session, items):
    async def aread_project(project_name):
        return session

    return NS(
        aread_project=aread_project,
        datasets=NS(experiment_runs=NS(query=lambda *a, **kw: _Pages(items))),
    )


def _session(dataset="d-1", metadata=None):
    return NS(
        id="s-1",
        reference_dataset_id=dataset,
        start_time=None,
        end_time=None,
        extra={"metadata": metadata or {}},
    )


def test_fetch_experiment_groups_dedups_and_drops_content(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    items = [
        NS(id="ex-1", inputs={"q": "private"}, runs=[_run("r1", 1.0), _run("r2", 0.0)]),
        NS(id="ex-2", inputs={"q": "private"}, runs=[_run("r1", 1.0), _run("r3", None)]),
    ]
    meta = {"git": {"commit": "abc", "dirty": False, "remote_url": "x"}, "dataset_version": "v"}
    src = LangSmithSource(project_name="p", client=_client(_session(metadata=meta), items))
    listing = src.fetch_experiment("exp")
    assert [r.run_id for r in listing.runs] == ["r1", "r2", "r3"]
    assert listing.runs[2].feedback[0].score is None
    assert listing.conditions["git_commit"] == "abc"
    assert "remote_url" not in str(listing.conditions)
    assert "private" not in str(listing) and "secret comment" not in str(listing)
    assert listing.complete


def test_fetch_experiment_requires_dataset_and_allowlist(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ALLOWED_PROJECTS", raising=False)
    src = LangSmithSource(project_name="p", client=_client(_session(dataset=None), []))
    with pytest.raises(ValueError, match="not an experiment"):
        src.fetch_experiment("exp")
    monkeypatch.setenv("LANGSMITH_ALLOWED_PROJECTS", "other-*")
    with pytest.raises(PermissionError):
        src.fetch_experiment("exp")
