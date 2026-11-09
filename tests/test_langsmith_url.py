"""Tests for LangSmith run URL parsing and UUIDv7 timestamps (pure, no network)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from self_improve_cli.sources.langsmith_url import (
    is_url,
    parse_langsmith_run_url,
    uuid7_timestamp,
)

ORG = "11111111-1111-4111-8111-111111111111"
PROJ = "22222222-2222-4222-8222-222222222222"
TRACE = "33333333-3333-4333-8333-333333333333"
RUN = "44444444-4444-4444-8444-444444444444"
BASE = f"https://eu.smith.example.com/o/{ORG}/projects/p/{PROJ}"


def test_is_url():
    assert is_url("https://smith.example.com/x")
    assert is_url("  HTTP://host/x")
    assert not is_url(RUN)
    assert not is_url("")


def test_parse_get_url_shape_with_start_time():
    """The shape returned by runs.get_url: /trace/<trace>/run/<run>?start_time=..."""
    ref = parse_langsmith_run_url(
        f"{BASE}/trace/{TRACE}/run/{RUN}?start_time=2026-11-06T16%3A36%3A49.988001%2B00%3A00"
    )
    assert ref.project_id == PROJ
    assert ref.trace_id == TRACE
    assert ref.run_id == RUN
    assert ref.start_time == datetime(2026, 11, 6, 16, 36, 49, 988001, tzinfo=UTC)


def test_parse_trace_url_without_run_points_at_root():
    ref = parse_langsmith_run_url(f"{BASE}/trace/{TRACE}")
    assert ref.run_id == TRACE
    assert ref.trace_id == TRACE


def test_parse_r_shape_with_query_trace_id():
    ref = parse_langsmith_run_url(
        f"{BASE}/r/{RUN}?trace_id={TRACE}&start_time=2026-11-06T16:36:49Z"
    )
    assert (ref.project_id, ref.trace_id, ref.run_id) == (PROJ, TRACE, RUN)
    assert ref.start_time == datetime(2026, 11, 6, 16, 36, 49, tzinfo=UTC)


def test_parse_peek_shape():
    ref = parse_langsmith_run_url(f"{BASE}?peek={RUN}&peeked_trace={TRACE}")
    assert (ref.project_id, ref.trace_id, ref.run_id) == (PROJ, TRACE, RUN)
    assert ref.start_time is None


def test_parse_missing_trace_keeps_none():
    ref = parse_langsmith_run_url(f"{BASE}/r/{RUN}")
    assert ref.run_id == RUN
    assert ref.trace_id is None
    assert ref.start_time is None


def test_parse_bad_start_time_is_ignored():
    ref = parse_langsmith_run_url(f"{BASE}/r/{RUN}?start_time=yesterday")
    assert ref.start_time is None


def test_parse_naive_start_time_is_utc():
    ref = parse_langsmith_run_url(f"{BASE}/r/{RUN}?start_time=2026-11-06T16:36:49")
    assert ref.start_time == datetime(2026, 11, 6, 16, 36, 49, tzinfo=UTC)


def test_parse_project_url_without_run_raises():
    with pytest.raises(ValueError, match="no run ID"):
        parse_langsmith_run_url(BASE)


def test_parse_non_langsmith_url_raises():
    with pytest.raises(ValueError, match="Not a LangSmith run URL"):
        parse_langsmith_run_url("https://example.com/projects/p/not-a-uuid/r/also-not")


@pytest.mark.parametrize("garbage", ["not a url", "ftp://host/x", "https://", ""])
def test_parse_garbage_raises(garbage):
    with pytest.raises(ValueError):
        parse_langsmith_run_url(garbage)


def test_parse_invalid_run_id_raises():
    with pytest.raises(ValueError, match="no run ID"):
        parse_langsmith_run_url(f"{BASE}/r/not-a-uuid")


def test_uuid7_timestamp():
    # 0x019ff0e57189 ms since the epoch
    expected = datetime.fromtimestamp(0x019FF0E57189 / 1000, tz=UTC)
    assert uuid7_timestamp("019ff0e5-7189-713d-8ab9-c032edf9d4dd") == expected
    assert expected.year == 2026


def test_uuid7_timestamp_non_v7_or_invalid():
    assert uuid7_timestamp(RUN) is None  # version 4
    assert uuid7_timestamp("run-abc") is None
