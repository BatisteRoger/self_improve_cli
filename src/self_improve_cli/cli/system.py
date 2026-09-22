"""System commands — skill discovery, init, doctor (not trace analysis)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_improve_cli.cli.common import EXIT_ERROR, EXIT_OK

# Single source of truth for the .env template. Used by `self-improve init`
# to generate a .env file. No separate .env.example file is shipped — this
# constant is the only copy, avoiding sync issues between root and package files.
ENV_TEMPLATE = """\
# Copy this file to .env and fill in your values.
# Never commit the real .env file.

# --- LangSmith (optional, only needed for the `fetch` command) ---

# LangSmith API key. Use a least-privileged key.
#
# This is the ONLY security control in this file. The API key's workspace
# and project scoping on the LangSmith server side is what actually prevents
# unauthorized access. Everything else below is convenience, not security.
LANGSMITH_API_KEY=""

# LangSmith API endpoint. Defaults to the US endpoint.
# LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

# Default LangSmith project name for list/fetch commands.
#
# This is a CONVENIENCE DEFAULT, not a security control. It can be overridden
# per-command with --project. You can leave it empty and always pass --project,
# or set it to your most-used project to avoid typing it every time.
LANGSMITH_PROJECT=""

# --- Project allowlist (optional safety net, NOT a security control) ---
#
# Comma-separated list of allowed project names or glob patterns.
# If empty/unset, all projects accessible to the API key are allowed.
# If set, only projects matching at least one pattern can be queried.
#
# This is a convenience safety net to prevent typos (e.g. accidentally
# typing "production" instead of "preprod"). It is NOT a security boundary -
# anyone with access to this .env file can edit it or bypass it with --project.
# Real access control is enforced by the API key's server-side scoping.
#
# Patterns use fnmatch syntax: * matches anything, ? matches one char.
#
# Examples:
#   LANGSMITH_ALLOWED_PROJECTS="staging*,my-agent-dev,my-agent-test"
#   LANGSMITH_ALLOWED_PROJECTS="my-agent-*"
#   LANGSMITH_ALLOWED_PROJECTS=""  # all projects allowed (rely on API key)
LANGSMITH_ALLOWED_PROJECTS=""

# --- Prompt Hub access (safety feature) ---
#
# Whether to allow the `prompt pull` command to download prompts from
# LangSmith Prompt Hub. Disabled by default - set to "true" to enable.
#
# This prevents an agent from reading potentially sensitive prompt logic
# without explicit user consent.
#
# Use --workspace <UUID> to pull from non-default workspaces (e.g. Flows).
ENABLE_PROMPT_HUB="false"

# --- Anonymizer backend (performance vs. coverage trade-off) ---
#
# Which backend to use for PII/secrets detection during `fetch`.
# Override per-command with --anonymizer.
#
#   auto      Use Presidio if installed, fall back to regex. Default.
#   presidio  Force Presidio. Errors if not installed (falls back to regex).
#   regex     Skip Presidio entirely. Faster, less comprehensive for PII.
#             Secrets are still fully detected (regex patterns are always run).
SELFIIMPROVE_ANONYMIZER="auto"
"""


def _find_skills_dir() -> Path | None:
    """Find the skills directory, checking repo root, .agents, and package data."""
    # 1. Repo root (for cloned repo users)
    root_skills = Path("skills")
    if root_skills.is_dir() and any(root_skills.glob("*/SKILL.md")):
        return root_skills
    # 2. .agents/skills (for users who installed via npx skills add)
    agents_skills = Path(".agents/skills")
    if agents_skills.is_dir() and any(agents_skills.glob("*/SKILL.md")):
        return agents_skills
    # 3. Package data (for pip-installed users)
    try:
        from importlib.resources import files

        pkg_skills = Path(str(files("self_improve_cli"))) / "skills"
        if pkg_skills.is_dir() and any(pkg_skills.glob("*/SKILL.md")):
            return pkg_skills
    except Exception:  # noqa: BLE001
        pass
    return None


def _cmd_skill(args: argparse.Namespace) -> int:
    """List available skills or print a specific skill's SKILL.md."""
    skills_dir = _find_skills_dir()
    if skills_dir is None:
        print(
            "No skills found. Install with: npx skills add BatisteRoger/self_improve_cli",
            file=sys.stderr,
        )
        return EXIT_ERROR

    skill_name = getattr(args, "skill_name", None)
    if skill_name is None:
        # List available skills
        skills = sorted(
            d.name for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
        )
        if not skills:
            print("No skills found.", file=sys.stderr)
            return EXIT_ERROR
        for name in skills:
            print(name)
        return EXIT_OK

    # Print a specific skill
    skill_path = skills_dir / skill_name / "SKILL.md"
    if not skill_path.exists():
        available = sorted(
            d.name for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
        )
        print(
            f"Skill '{skill_name}' not found. Available: {', '.join(available)}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    print(skill_path.read_text(encoding="utf-8"))
    return EXIT_OK


def _cmd_init(args: argparse.Namespace) -> int:
    """Create a .env file from the built-in template if it doesn't exist."""
    env_path = Path(".env")
    if env_path.exists():
        print(".env already exists. Edit it to fill in your values.", file=sys.stderr)
        return EXIT_ERROR
    env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
    print("Created .env. Edit it to fill in your LangSmith API key.")
    return EXIT_OK


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Run a read-only local setup diagnostic."""
    from self_improve_cli.cli.doctor import format_report_markdown, run_doctor

    report = run_doctor(profile=args.profile, offline=args.offline)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report_markdown(report))
    return EXIT_OK if not report.has_failures else EXIT_ERROR


def register(sub: argparse._SubParsersAction) -> None:
    """Register system commands on the given subparsers action."""
    # skill
    p = sub.add_parser("skill", help="List available skills or print a specific skill")
    p.add_argument("skill_name", nargs="?", default=None, help="Skill name (omit to list)")
    p.set_defaults(func=_cmd_skill)

    # init
    p = sub.add_parser("init", help="Create a .env file from the built-in template")
    p.set_defaults(func=_cmd_init)

    # doctor
    p = sub.add_parser(
        "doctor",
        help="Read-only local setup diagnostic (no installs, no network)",
    )
    p.add_argument(
        "--profile",
        choices=["default", "fetch"],
        default="default",
        help="Check profile: 'default' for analysis commands, 'fetch' escalates "
        "langsmith_extra and env_api_key to fail (default: default)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Skip checks requiring Docker, databases, or network (no-op for this "
        "project — all checks are already local-only)",
    )
    p.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p.set_defaults(func=_cmd_doctor)
