"""Shared CLI plumbing — exit codes, store construction, output formatting.

Command modules import these helpers; they are deliberately tiny so the
command files stay focused on argument handling and output shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_improve_cli.storage import TraceStore

# Exit codes
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def add_common_opts(p: argparse.ArgumentParser) -> None:
    """Options shared by most commands: --format and --data-dir."""
    p.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p.add_argument("--data-dir", default=None, help="Local data root (default: ./data)")


def _get_store(args: argparse.Namespace) -> TraceStore:
    data_root = Path(args.data_dir) if args.data_dir else Path("data")
    return TraceStore(data_root=data_root, keep_raw=args.keep_raw)


def _output(data: Any, args: argparse.Namespace) -> None:
    """Print results in the requested format (markdown or json)."""
    if args.format == "json":
        if isinstance(data, str):
            print(json.dumps({"content": data}, ensure_ascii=False))
        else:
            print(json.dumps(data, indent=2, default=str, ensure_ascii=False))
    else:
        if isinstance(data, str):
            print(data)
        else:
            print(json.dumps(data, indent=2, default=str, ensure_ascii=False))
