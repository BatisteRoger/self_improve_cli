"""Structural redaction: mapping-table value replacement and key dropping.

Some target agents pseudonymize their own data and record the
re-identification table inside the trace (a list of ``{value, token}``
entries), while the raw text still appears elsewhere — the root input, runs
that execute before the pseudonymization step. Pattern detection cannot
know those values. This module uses the table itself:

1. ``collect_value_map`` reads every configured table, at any depth, into a
   ``{value: token}`` mapping.
2. ``replace_values`` replaces each occurrence of a value, in any letter
   case, by its token across the whole payload, so the target's own tag
   vocabulary stays consistent.
3. ``drop_keys`` replaces configured keys (and every table key) with a
   ``<dropped:N chars>`` marker, so sizes stay visible.

Configuration is generic and comes from the caller — no key name is
hardcoded here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from self_improve_cli.domain import Run

_MAX_DEPTH = 50


@dataclass(frozen=True)
class ValueMapSpec:
    """Where a mapping table lives: ``<table_key>:<value_field>:<token_field>``."""

    table_key: str
    value_field: str
    token_field: str


def parse_drop_keys(raw: str) -> frozenset[str]:
    """Parse a comma-separated list of key names."""
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def parse_value_maps(raw: str) -> tuple[ValueMapSpec, ...]:
    """Parse comma-separated ``<table_key>:<value_field>:<token_field>`` specs.

    Raises ValueError on a malformed spec: a privacy setting that is
    silently ignored would let raw values through.
    """
    specs: list[ValueMapSpec] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        parts = [p.strip() for p in item.split(":")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"invalid value-map spec {item.strip()!r}: expected "
                "<table_key>:<value_field>:<token_field>"
            )
        specs.append(ValueMapSpec(*parts))
    return tuple(specs)


def _walk(value: Any, depth: int = 0) -> Iterator[dict[str, Any]]:
    """Yield every dict nested in a JSON-like value."""
    if depth > _MAX_DEPTH:
        return
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _walk(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v, depth + 1)


def _run_payloads(run: Run) -> Iterator[Any]:
    yield run.inputs
    yield run.outputs
    yield run.extra
    for msg in [*run.input_messages, run.output_message]:
        if msg is not None:
            for tc in msg.tool_calls:
                yield tc.args


def collect_value_map(runs: Iterable[Run], specs: Iterable[ValueMapSpec]) -> dict[str, str]:
    """Collect ``{value: token}`` pairs from every configured table in the runs."""
    specs = tuple(specs)
    mapping: dict[str, str] = {}
    if not specs:
        return mapping
    for run in runs:
        for payload in _run_payloads(run):
            for d in _walk(payload):
                for spec in specs:
                    table = d.get(spec.table_key)
                    if not isinstance(table, list):
                        continue
                    for entry in table:
                        if not isinstance(entry, dict):
                            continue
                        value = entry.get(spec.value_field)
                        token = entry.get(spec.token_field)
                        if isinstance(value, str) and value and isinstance(token, str):
                            mapping.setdefault(value, token)
    return mapping


def _map_strings(value: Any, fn: Callable[[str], str], depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return value
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_strings(v, fn, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_strings(v, fn, depth + 1) for v in value]
    return value


def _apply_to_run(run: Run, fn: Callable[[str], str]) -> None:
    run.error = fn(run.error) if run.error else run.error
    run.inputs = _map_strings(run.inputs, fn)
    run.outputs = _map_strings(run.outputs, fn)
    run.extra = _map_strings(run.extra, fn)
    for msg in [*run.input_messages, run.output_message]:
        if msg is not None:
            msg.text = fn(msg.text)
            for tc in msg.tool_calls:
                tc.args = _map_strings(tc.args, fn)


def replace_values(runs: Iterable[Run], mapping: dict[str, str]) -> int:
    """Replace every occurrence of a mapped value by its token, ignoring case.

    ``DUPONT`` and ``dupont`` get the token of ``Dupont``. Longer values win
    over values they contain. Returns the replacement count.
    """
    if not mapping:
        return 0
    by_lower: dict[str, str] = {}
    for value, token in mapping.items():
        by_lower.setdefault(value.lower(), token)
    pattern = re.compile(
        "|".join(re.escape(v) for v in sorted(mapping, key=len, reverse=True)), re.IGNORECASE
    )
    count = 0

    def token_for(match: re.Match[str]) -> str:
        # Unicode case pairs where lower() and IGNORECASE disagree still get
        # masked: never return the matched raw text.
        return by_lower.get(match.group(0).lower(), "<MAPPED_VALUE>")

    def substitute(text: str) -> str:
        nonlocal count
        result, n = pattern.subn(token_for, text)
        count += n
        return result

    for run in runs:
        _apply_to_run(run, substitute)
    return count


def _dropped_marker(value: Any) -> str:
    size = (
        len(value)
        if isinstance(value, str)
        else len(json.dumps(value, ensure_ascii=False, default=str))
    )
    return f"<dropped:{size} chars>"


def _drop(value: Any, keys: frozenset[str], counter: list[int], depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in keys:
                out[k] = _dropped_marker(v)
                counter[0] += 1
            else:
                out[k] = _drop(v, keys, counter, depth + 1)
        return out
    if isinstance(value, list):
        return [_drop(v, keys, counter, depth + 1) for v in value]
    return value


def drop_keys(runs: Iterable[Run], keys: frozenset[str]) -> int:
    """Replace each subtree under a key in ``keys`` with a size marker.

    Applies at any depth of run inputs, outputs, extra, and tool call args.
    Returns the number of dropped subtrees.
    """
    if not keys:
        return 0
    counter = [0]
    for run in runs:
        run.inputs = _drop(run.inputs, keys, counter)
        run.outputs = _drop(run.outputs, keys, counter)
        run.extra = _drop(run.extra, keys, counter)
        for msg in [*run.input_messages, run.output_message]:
            if msg is not None:
                for tc in msg.tool_calls:
                    tc.args = _drop(tc.args, keys, counter)
    return counter[0]
