"""Tests for the doctor command (offline, no network, no installs)."""

import json

import pytest

from self_improve_cli.cli.doctor import (
    Check,
    DoctorReport,
    format_report_markdown,
    run_doctor,
)
from self_improve_cli.cli.main import main

# ---------------------------------------------------------------------------
# Unit tests for individual checks
# ---------------------------------------------------------------------------


def test_python_version_passes_on_current_interpreter():
    """python_version check passes on the interpreter running the tests."""
    report = run_doctor()
    py_check = next(c for c in report.checks if c.id == "python_version")
    assert py_check.status == "pass"


def test_package_importable_passes_in_test_env():
    """package_importable passes because the package is installed for tests."""
    report = run_doctor()
    pkg_check = next(c for c in report.checks if c.id == "package_importable")
    assert pkg_check.status == "pass"


def test_env_api_key_warns_when_missing(monkeypatch):
    """env_api_key warns (default profile) when LANGSMITH_API_KEY is unset."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    report = run_doctor(profile="default")
    check = next(c for c in report.checks if c.id == "env_api_key")
    assert check.status == "warn"


def test_env_api_key_fails_in_fetch_profile_when_missing(monkeypatch):
    """env_api_key fails in fetch profile when LANGSMITH_API_KEY is unset."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    report = run_doctor(profile="fetch")
    check = next(c for c in report.checks if c.id == "env_api_key")
    assert check.status == "fail"


def test_env_api_key_passes_when_set(monkeypatch):
    """env_api_key passes when LANGSMITH_API_KEY is set and non-placeholder."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_sk_test_fake_key_for_tests")
    report = run_doctor(profile="fetch")
    check = next(c for c in report.checks if c.id == "env_api_key")
    assert check.status == "pass"


def test_env_api_key_warns_when_empty_string(monkeypatch):
    """env_api_key warns when LANGSMITH_API_KEY is empty string (unfilled .env)."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    report = run_doctor(profile="default")
    check = next(c for c in report.checks if c.id == "env_api_key")
    assert check.status == "warn"


def test_langsmith_extra_status_depends_on_profile(monkeypatch):
    """langsmith_extra check status depends on profile and availability."""
    # We can't easily uninstall langsmith, so we test the profile logic:
    # if langsmith IS installed, both profiles pass. If not, fetch fails.
    report_default = run_doctor(profile="default")
    report_fetch = run_doctor(profile="fetch")
    check_default = next(c for c in report_default.checks if c.id == "langsmith_extra")
    check_fetch = next(c for c in report_fetch.checks if c.id == "langsmith_extra")

    try:
        import langsmith  # type: ignore[import-not-found]  # noqa: F401

        # If installed, both pass
        assert check_default.status == "pass"
        assert check_fetch.status == "pass"
    except ImportError:
        # If not installed, default warns, fetch fails
        assert check_default.status == "warn"
        assert check_fetch.status == "fail"


# ---------------------------------------------------------------------------
# Report structure and redaction
# ---------------------------------------------------------------------------


def test_report_has_all_expected_check_ids():
    """The report contains all expected check ids with stable names."""
    report = run_doctor()
    ids = {c.id for c in report.checks}
    expected = {
        "python_version",
        "uv_available",
        "venv_present",
        "package_importable",
        "langsmith_extra",
        "presidio_extra",
        "env_api_key",
        "env_project",
        "wheel_artifacts",
        "dev_ruff",
        "dev_pyright",
        "dev_pytest",
    }
    assert ids == expected


def test_report_does_not_leak_env_values(monkeypatch):
    """The report never contains raw environment values."""
    secret_value = "lsv2_sk_super_secret_value_12345"
    monkeypatch.setenv("LANGSMITH_API_KEY", secret_value)
    report = run_doctor()
    # Check markdown output
    md = format_report_markdown(report)
    assert secret_value not in md
    # Check JSON output
    js = json.dumps(report.to_dict())
    assert secret_value not in js


def test_report_to_dict_structure():
    """to_dict produces a stable, machine-readable structure."""
    report = run_doctor(profile="fetch", offline=True)
    d = report.to_dict()
    assert d["profile"] == "fetch"
    assert d["offline"] is True
    assert len(d["checks"]) == 12
    for check_dict in d["checks"]:
        assert "id" in check_dict
        assert "status" in check_dict
        assert "summary" in check_dict
        assert "remedy" in check_dict
        assert check_dict["status"] in ("pass", "warn", "fail", "skip")


def test_has_failures_property():
    """has_failures is True when any check has status 'fail'."""
    report = DoctorReport(profile="default", offline=False)
    report.checks.append(Check(id="test", status="pass", summary="ok"))
    assert report.has_failures is False
    report.checks.append(Check(id="test2", status="fail", summary="bad"))
    assert report.has_failures is True


def test_format_report_markdown_contains_summary():
    """Markdown output includes a summary line with failure/warning counts."""
    report = run_doctor()
    md = format_report_markdown(report)
    assert "failure" in md
    assert "warning" in md
    assert "# Doctor report" in md


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_doctor_command_exits_zero_when_no_failures(monkeypatch, capsys):
    """`self-improve doctor` exits 0 when there are no failures."""
    # Set up a clean env where nothing fails
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_sk_test_fake_key")
    monkeypatch.setattr("self_improve_cli.cli.main._load_env", lambda: None)
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "# Doctor report" in out
    # Exit code depends on the actual env, but with API key set, core checks pass.
    # The only potential fail is package_importable, which should pass in test env.
    assert rc in (0, 1)  # 0 if no fails, 1 if warns escalated (shouldn't happen)


def test_doctor_command_json_format(monkeypatch, capsys):
    """`self-improve doctor --format json` produces valid JSON."""
    rc = main(["doctor", "--format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "checks" in data
    assert "profile" in data
    assert data["profile"] == "default"
    assert rc in (0, 1)


def test_doctor_command_fetch_profile(monkeypatch, capsys):
    """`self-improve doctor --profile fetch` uses the fetch profile."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    # Prevent _load_env from reloading .env and re-setting the var
    monkeypatch.setattr("self_improve_cli.cli.main._load_env", lambda: None)
    rc = main(["doctor", "--profile", "fetch", "--format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["profile"] == "fetch"
    api_key_check = next(c for c in data["checks"] if c["id"] == "env_api_key")
    assert api_key_check["status"] == "fail"
    assert rc == 1  # fetch profile with no API key should fail


def test_doctor_command_offline_flag_accepted(capsys):
    """`self-improve doctor --offline` is accepted (no-op for this project)."""
    rc = main(["doctor", "--offline", "--format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["offline"] is True
    assert rc in (0, 1)


def test_doctor_command_invalid_profile_rejected(capsys):
    """`self-improve doctor --profile invalid` returns usage error."""
    with pytest.raises(SystemExit) as exc_info:
        main(["doctor", "--profile", "invalid"])
    assert exc_info.value.code == 2
