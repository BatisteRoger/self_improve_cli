"""Tests for structural redaction (mapping-table replacement, key dropping),
the French regex recognizers, and the privacy env-var resolution.

All data is synthetic.
"""

import argparse
import json

import pytest

from self_improve_cli.cli.discovery import _resolve_anonymizer, _resolve_structural_policy
from self_improve_cli.cli.main import main
from self_improve_cli.domain import Message, RunType, ToolCall, Trace
from self_improve_cli.privacy import (
    ValueMapSpec,
    anonymize_trace,
    parse_value_maps,
)
from self_improve_cli.storage import TraceStore
from tests.helpers import make_root, make_run

_SPEC = ValueMapSpec("pii_table", "original", "placeholder")
_TABLE = [
    {"placeholder": "<NAME_1>", "original": "Alice Martin", "kind": "PERSON"},
    {"placeholder": "<NAME_2>", "original": "Alice", "kind": "PERSON"},
    {"placeholder": "<CITY_1>", "original": "Springfield", "kind": "LOCATION"},
]


def _pseudonymized_trace() -> Trace:
    """Root input carries raw text; the table sits in the root output; a
    pre-pseudonymization child run sees raw text in inputs, messages, and
    tool call args."""
    root = make_root()
    root.inputs = {"messages": [{"role": "user", "content": "I am Alice Martin from Springfield"}]}
    root.outputs = {
        "answer": "Hello <NAME_1>",
        "pii_table": _TABLE,
        "raw_turns": ["I am Alice Martin from Springfield"],
    }
    early = make_run(
        "1",
        run_type=RunType.LLM,
        inputs={"state": {"history": [{"text": "Alice Martin, Springfield"}]}},
        input_messages=[Message(role="human", text="Alice moved to Springfield")],
        output_message=Message(
            role="ai",
            tool_calls=[ToolCall(id="c1", name="lookup", args={"who": "Alice Martin"})],
        ),
    )
    return Trace(trace_id="t1", runs=[root, early])


def test_mapping_table_in_root_output_masks_raw_text_everywhere():
    trace, report = anonymize_trace(_pseudonymized_trace(), backend="regex", value_maps=(_SPEC,))
    dumped = json.dumps(
        [[r.inputs, r.outputs, r.extra, [m.text for m in r.input_messages]] for r in trace.runs]
        + [trace.runs[1].output_message.tool_calls[0].args]  # type: ignore[union-attr]
    )
    for raw in ("Alice", "Martin", "Springfield"):
        assert raw not in dumped
    # The target's own tag vocabulary is reused, and the longer value wins.
    assert trace.runs[0].inputs["messages"][0]["content"] == "I am <NAME_1> from <CITY_1>"
    assert trace.runs[1].input_messages[0].text == "<NAME_2> moved to <CITY_1>"
    assert trace.runs[1].output_message.tool_calls[0].args == {"who": "<NAME_1>"}  # type: ignore[union-attr]
    # The table key is dropped even when not listed in drop_keys.
    assert trace.runs[0].outputs["pii_table"].startswith("<dropped:")
    assert report.entity_counts["MAPPED_VALUE"] >= 6
    assert report.entity_counts["DROPPED_KEY"] == 1


def test_mapping_table_nested_in_child_run_applies_to_whole_trace():
    root = make_root()
    root.inputs = {"question": "Where does Bob Stone live?"}
    child = make_run(
        "1",
        run_type=RunType.CHAIN,
        outputs={"state": {"steps": [{"meta": {"pii_table": [_TABLE[0]]}}]}},
    )
    child.outputs["state"]["steps"][0]["meta"]["pii_table"].append(
        {"placeholder": "<NAME_9>", "original": "Bob Stone"}
    )
    trace, _ = anonymize_trace(Trace(trace_id="t1", runs=[root, child]), value_maps=(_SPEC,))
    assert trace.runs[0].inputs["question"] == "Where does <NAME_9> live?"
    assert "Bob" not in json.dumps(trace.runs[1].outputs)


def test_drop_keys_replaces_subtree_with_size_marker_at_any_depth():
    secret = {"text": "raw content", "n": 1}
    root = make_root()
    root.extra = {"metadata": {"items": [{"keep": "ok", "raw_blob": secret}]}}
    trace, report = anonymize_trace(
        Trace(trace_id="t1", runs=[root]), backend="regex", drop_keys=frozenset({"raw_blob"})
    )
    item = trace.runs[0].extra["metadata"]["items"][0]
    assert item["keep"] == "ok"
    assert item["raw_blob"] == f"<dropped:{len(json.dumps(secret))} chars>"
    assert report.entity_counts["DROPPED_KEY"] == 1


@pytest.mark.parametrize("raw", ["", "pii_table", "a:b", "a::c", "a:b:c:d"])
def test_malformed_value_map_spec_fails_closed(raw):
    if raw == "":
        assert parse_value_maps(raw) == ()
        return
    with pytest.raises(ValueError, match="value-map spec"):
        parse_value_maps(raw)


# ---------------------------------------------------------------------------
# French regex recognizers
# ---------------------------------------------------------------------------


def _scan(text: str) -> tuple[str, dict[str, int]]:
    trace = Trace(trace_id="t1", runs=[make_root(), make_run("1", inputs={"t": text})])
    out, report = anonymize_trace(trace, backend="regex")
    return out.runs[1].inputs["t"], report.entity_counts


@pytest.mark.parametrize(
    "phone", ["06 12 34 56 78", "01.23.45.67.89", "07-98-76-54-32", "0612345678"]
)
def test_fr_phone_detected(phone):
    text, counts = _scan(f"Appelez-moi au {phone} demain.")
    assert phone not in text
    assert counts.get("PHONE") == 1


@pytest.mark.parametrize("nir", ["1 85 05 78 006 084 36", "285057800608436", "2 85 05 2A 006 084"])
def test_fr_nir_detected(nir):
    text, counts = _scan(f"Mon numero de securite sociale est {nir}.")
    assert nir not in text
    assert counts.get("FR_NIR") == 1
    assert "PHONE" not in counts


def test_birth_date_only_in_context():
    text, counts = _scan("Elle est née le 12/03/1985. Echeance au 12/03/2026, taux 0.25.")
    assert "12/03/1985" not in text
    assert "Echeance au 12/03/2026, taux 0.25." in text
    assert counts.get("BIRTH_DATE") == 1


def test_fr_patterns_spare_financial_text():
    text = "Capital 150000 EUR, frais 0.85 %, versement 2518.17, ref 2024-05-12."
    out, counts = _scan(text)
    assert out == text
    assert counts == {}


# ---------------------------------------------------------------------------
# Env var resolution
# ---------------------------------------------------------------------------


_NAMES = ("ANONYMIZER", "DROP_KEYS", "VALUE_MAP")


@pytest.fixture
def clean_env(monkeypatch):
    for name in _NAMES:
        monkeypatch.delenv(f"SELF_IMPROVE_{name}", raising=False)
        monkeypatch.delenv(f"SELFIIMPROVE_{name}", raising=False)
    return monkeypatch


def test_correct_env_prefix_wins_over_legacy(clean_env):
    clean_env.setenv("SELF_IMPROVE_ANONYMIZER", "regex")
    clean_env.setenv("SELFIIMPROVE_ANONYMIZER", "presidio")
    assert _resolve_anonymizer(argparse.Namespace(anonymizer=None)) == "regex"


def test_legacy_misspelled_env_still_read(clean_env):
    clean_env.setenv("SELFIIMPROVE_ANONYMIZER", "presidio")
    clean_env.setenv("SELFIIMPROVE_DROP_KEYS", "a, b")
    clean_env.setenv("SELFIIMPROVE_VALUE_MAP", "t:v:k")
    assert _resolve_anonymizer(argparse.Namespace(anonymizer=None)) == "presidio"
    assert _resolve_structural_policy() == (frozenset({"a", "b"}), (ValueMapSpec("t", "v", "k"),))


class _TableSource:
    def __init__(self, project_name=None, **_kwargs):
        self.project_name = project_name

    def fetch_trace(self, trace_id, project_id=None):
        trace = _pseudonymized_trace()
        trace.trace_id = trace_id
        for run in trace.runs:
            run.trace_id = trace_id
        return trace

    def fetch_feedback(self, trace_id):
        return []


def test_fetch_applies_env_structural_policy_before_persisting(tmp_path, capsys, clean_env):
    import self_improve_cli.sources.langsmith as ls_module

    clean_env.setattr("self_improve_cli.cli.main._load_env", lambda: None)
    clean_env.setattr(ls_module, "LangSmithSource", _TableSource)
    clean_env.setenv("SELF_IMPROVE_VALUE_MAP", "pii_table:original:placeholder")
    clean_env.setenv("SELF_IMPROVE_DROP_KEYS", "raw_turns")

    rc = main(["fetch", "t1", "--data-dir", str(tmp_path), "--anonymizer", "regex"])
    assert rc == 0
    capsys.readouterr()
    on_disk = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file())
    assert "Alice" not in on_disk
    assert "Springfield" not in on_disk
    assert "<NAME_1>" in on_disk
    stored = TraceStore(tmp_path).load_trace("t1")
    assert stored.runs[0].outputs["raw_turns"].startswith("<dropped:")


def test_fetch_rejects_malformed_value_map_before_network(tmp_path, capsys, clean_env):
    import self_improve_cli.sources.langsmith as ls_module

    def _no_network(*_a, **_k):
        raise AssertionError("source must not be built on a bad privacy config")

    clean_env.setattr("self_improve_cli.cli.main._load_env", lambda: None)
    clean_env.setattr(ls_module, "LangSmithSource", _no_network)
    clean_env.setenv("SELF_IMPROVE_VALUE_MAP", "pii_table")
    rc = main(["fetch", "t1", "--data-dir", str(tmp_path)])
    assert rc != 0
    assert "value-map spec" in capsys.readouterr().err
