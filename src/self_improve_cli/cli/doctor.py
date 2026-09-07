"""Doctor: read-only local setup diagnostic.

Inspects the local environment and reports what's ready, missing, or
misconfigured — without fixing anything. Designed for coding agents
onboarding to this repo: a green report means "the local setup is
consistent enough to build, run, and test the CLI."

Read-only by construction: no dependency installs, no service starts,
no network calls, no file generation. If a check would need to change
something to succeed, it reports the gap instead.

No secret leakage: environment values are never echoed. Only presence
and non-placeholder status is reported.

Profiles:
- default: checks needed to run analysis commands (skeleton, narrative,
  metrics) on already-fetched traces.
- fetch: escalates langsmith_extra and env_api_key to fail, since the
  `fetch` command cannot work without them.

Exit codes: 0 = no failures, 1 = one or more failed checks, 2 = usage error.

What is NOT checked:
- Whether LANGSMITH_API_KEY is valid or reachable (that would require a
  network call and a paid API).
- Whether .venv dependencies are up-to-date (that would require an install).
- Freshness of generated artifacts (skills/, py.typed are checked for
  existence only).
- Docker, databases, or other services (this project has none).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CheckStatus = Literal["pass", "warn", "fail", "skip"]

# Placeholder values that indicate an unfilled .env (from .env.example).
_PLACEHOLDER_VALUES = {""}

# Requires-python from pyproject.toml — kept in sync manually.
_MIN_PYTHON = (3, 12)


@dataclass
class Check:
    """A single doctor check result.

    The id is stable across sessions — agents and developers reference
    checks by name. Renaming ids breaks that continuity.
    """

    id: str
    status: CheckStatus
    summary: str
    remedy: str | None = None
    profile: str = "default"


@dataclass
class DoctorReport:
    """Aggregated doctor report."""

    profile: str
    offline: bool
    checks: list[Check] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "offline": self.offline,
            "checks": [
                {
                    "id": c.id,
                    "status": c.status,
                    "summary": c.summary,
                    "remedy": c.remedy,
                }
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_python_version() -> Check:
    current = (sys.version_info.major, sys.version_info.minor)
    if current >= _MIN_PYTHON:
        return Check(
            id="python_version",
            status="pass",
            summary=f"Python {current[0]}.{current[1]} >= {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}",
        )
    return Check(
        id="python_version",
        status="fail",
        summary=(
            f"Python {current[0]}.{current[1]} is below the required "
            f"{_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}"
        ),
        remedy=f"Install Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ and recreate the venv.",
    )


def _check_uv_available() -> Check:
    uv_path = shutil.which("uv")
    if uv_path:
        return Check(id="uv_available", status="pass", summary="uv found on PATH")
    return Check(
        id="uv_available",
        status="warn",
        summary="uv not found on PATH",
        remedy="Install uv: https://docs.astral.sh/uv/ — or use pip/venv manually.",
    )


def _check_venv_present() -> Check:
    if Path(".venv").is_dir():
        return Check(id="venv_present", status="pass", summary=".venv/ exists")
    return Check(
        id="venv_present",
        status="warn",
        summary=".venv/ not found in cwd",
        remedy="Run `uv sync` to create the virtualenv and install dependencies.",
    )


def _check_package_importable() -> Check:
    try:
        import self_improve_cli  # noqa: F401

        return Check(
            id="package_importable",
            status="pass",
            summary="self_improve_cli is importable from the current interpreter",
        )
    except ImportError:
        return Check(
            id="package_importable",
            status="fail",
            summary="self_improve_cli is not importable — package not installed in this env",
            remedy="Run `uv sync` (or `pip install -e .`) in the project root.",
        )


def _check_langsmith_extra(profile: str) -> Check:
    try:
        import langsmith  # type: ignore[import-not-found]  # noqa: F401

        return Check(
            id="langsmith_extra",
            status="pass",
            summary="langsmith package is installed (enables `fetch`, `list`, `list-projects`)",
            profile=profile,
        )
    except ImportError:
        status: CheckStatus = "fail" if profile == "fetch" else "warn"
        return Check(
            id="langsmith_extra",
            status=status,
            summary="langsmith package not installed — `fetch`/`list` commands unavailable",
            remedy="Run `uv sync --extra langsmith` to install it.",
            profile=profile,
        )


def _check_presidio_extra() -> Check:
    try:
        import presidio_analyzer  # type: ignore[import-not-found]  # noqa: F401

        return Check(
            id="presidio_extra",
            status="pass",
            summary="presidio_analyzer installed (enhanced PII detection)",
        )
    except ImportError:
        return Check(
            id="presidio_extra",
            status="warn",
            summary="presidio_analyzer not installed — falling back to regex-only anonymization",
            remedy="Run `uv sync --extra privacy` for enhanced PII detection. "
            "Regex fallback is functional but less comprehensive.",
        )


def _is_placeholder(value: str | None) -> bool:
    """Check if an env value is empty or a known placeholder."""
    if value is None:
        return True
    return value.strip() in _PLACEHOLDER_VALUES


def _check_env_api_key(profile: str) -> Check:
    value = os.environ.get("LANGSMITH_API_KEY")
    if not _is_placeholder(value):
        return Check(
            id="env_api_key",
            status="pass",
            summary="LANGSMITH_API_KEY is set and non-placeholder (validity not checked)",
            profile=profile,
        )
    status: CheckStatus = "fail" if profile == "fetch" else "warn"
    return Check(
        id="env_api_key",
        status=status,
        summary="LANGSMITH_API_KEY is missing or empty",
        remedy="Set LANGSMITH_API_KEY in .env. Use a least-privileged key. "
        "Run `self-improve init` to create .env from .env.example.",
        profile=profile,
    )


def _check_env_project() -> Check:
    value = os.environ.get("LANGSMITH_PROJECT")
    if not _is_placeholder(value):
        return Check(
            id="env_project",
            status="pass",
            summary="LANGSMITH_PROJECT is set (convenience default for list/fetch)",
        )
    return Check(
        id="env_project",
        status="warn",
        summary="LANGSMITH_PROJECT is not set — pass --project per-command or set it in .env",
        remedy="Optional. Set LANGSMITH_PROJECT in .env to avoid passing --project every time.",
    )


def _check_wheel_artifacts() -> Check:
    """Check that force-included wheel artifacts exist."""
    missing: list[str] = []
    if not Path("skills").is_dir():
        missing.append("skills/")
    elif not any(Path("skills").glob("*/SKILL.md")):
        missing.append("skills/ (no SKILL.md files found)")
    if not Path("src/self_improve_cli/py.typed").exists():
        missing.append("src/self_improve_cli/py.typed")

    if not missing:
        return Check(
            id="wheel_artifacts",
            status="pass",
            summary="Wheel force-included artifacts present (skills/, py.typed)",
        )
    return Check(
        id="wheel_artifacts",
        status="warn",
        summary=f"Missing wheel artifacts: {', '.join(missing)}",
        remedy="These are included in the wheel build. Clone the repo or reinstall from source.",
    )


def _check_dev_ruff() -> Check:
    if shutil.which("ruff"):
        return Check(id="dev_ruff", status="pass", summary="ruff found on PATH")
    return Check(
        id="dev_ruff",
        status="warn",
        summary="ruff not found on PATH",
        remedy="Run `uv sync --extra dev` to install dev tooling.",
    )


def _check_dev_pyright() -> Check:
    if shutil.which("pyright"):
        return Check(id="dev_pyright", status="pass", summary="pyright found on PATH")
    return Check(
        id="dev_pyright",
        status="warn",
        summary="pyright not found on PATH",
        remedy="Run `uv sync --extra dev` to install dev tooling.",
    )


def _check_dev_pytest() -> Check:
    try:
        import pytest  # type: ignore[import-not-found]  # noqa: F401

        return Check(id="dev_pytest", status="pass", summary="pytest is importable")
    except ImportError:
        return Check(
            id="dev_pytest",
            status="warn",
            summary="pytest not installed",
            remedy="Run `uv sync --extra dev` to install dev tooling.",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_doctor(profile: str = "default", offline: bool = False) -> DoctorReport:
    """Run all doctor checks for the given profile.

    Read-only: no installs, no network, no mutations.
    """
    report = DoctorReport(profile=profile, offline=offline)

    # Core checks (always run)
    report.checks.append(_check_python_version())
    report.checks.append(_check_uv_available())
    report.checks.append(_check_venv_present())
    report.checks.append(_check_package_importable())
    report.checks.append(_check_langsmith_extra(profile))
    report.checks.append(_check_presidio_extra())
    report.checks.append(_check_env_api_key(profile))
    report.checks.append(_check_env_project())
    report.checks.append(_check_wheel_artifacts())

    # Dev tooling (always run, warn-only)
    report.checks.append(_check_dev_ruff())
    report.checks.append(_check_dev_pyright())
    report.checks.append(_check_dev_pytest())

    return report


def format_report_markdown(report: DoctorReport) -> str:
    """Format the report as compact Markdown for agent consumption."""
    lines = [
        f"# Doctor report (profile: {report.profile})",
        "",
        "| Status | Check | Summary |",
        "| --- | --- | --- |",
    ]
    for c in report.checks:
        icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}[c.status]
        lines.append(f"| {icon} | {c.id} | {c.summary} |")

    failures = [c for c in report.checks if c.status == "fail"]
    warnings = [c for c in report.checks if c.status == "warn"]
    lines.append("")
    lines.append(f"**{len(failures)} failure(s), {len(warnings)} warning(s)**")

    if failures:
        lines.append("")
        lines.append("## Failures")
        for c in failures:
            lines.append(f"- **{c.id}**: {c.summary}")
            if c.remedy:
                lines.append(f"  - Remedy: {c.remedy}")

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for c in warnings:
            lines.append(f"- **{c.id}**: {c.summary}")
            if c.remedy:
                lines.append(f"  - Remedy: {c.remedy}")

    return "\n".join(lines)
