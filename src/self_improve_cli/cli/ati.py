"""ATI commands — list and show target-agent architecture documents."""

from __future__ import annotations

import argparse
import json
import sys

from self_improve_cli.cli.common import (
    EXIT_ERROR,
    EXIT_OK,
    _get_store,
    _output,
    add_common_opts,
)


def _cmd_ati_list(args: argparse.Namespace) -> int:
    """List registered ATIs."""
    store = _get_store(args)
    atis = store.list_atis()
    if args.format == "json":
        print(json.dumps(atis, ensure_ascii=False))
    else:
        if not atis:
            print("No ATIs registered. Use the document-ati skill to create one.")
            return EXIT_OK
        for name in atis:
            print(name)
    return EXIT_OK


def _cmd_ati_show(args: argparse.Namespace) -> int:
    """Show an ATI's architecture document."""
    store = _get_store(args)
    content = store.load_ati(args.name)
    if content is None:
        print(
            f"ATI '{args.name}' not found. Create it with the document-ati skill.", file=sys.stderr
        )
        return EXIT_ERROR
    _output(content, args)
    return EXIT_OK


def register(sub: argparse._SubParsersAction) -> None:
    """Register ATI commands on the given subparsers action."""
    # ati
    p = sub.add_parser("ati", help="List or show target agent architecture docs")
    ati_sub = p.add_subparsers(dest="ati_command", required=True)

    p_ati_list = ati_sub.add_parser("list", help="List registered ATIs")
    add_common_opts(p_ati_list)
    p_ati_list.set_defaults(func=_cmd_ati_list)

    p_ati_show = ati_sub.add_parser("show", help="Show an ATI's architecture document")
    p_ati_show.add_argument("name", help="ATI name")
    add_common_opts(p_ati_show)
    p_ati_show.set_defaults(func=_cmd_ati_show)
