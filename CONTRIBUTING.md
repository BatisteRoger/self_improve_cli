# Contributing

Thank you for your interest in contributing. This project is designed to be safe, simple, and useful for both AI agents and humans.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and running commands.

```bash
# Clone and sync dependencies (creates a virtualenv automatically)
git clone <repo-url>
cd self_improve_cli
uv sync

# Optional: install privacy dependencies for Presidio-backed anonymization
uv sync --extra privacy

# Optional: install LangSmith support
uv sync --extra langsmith

# Optional: install everything (dev + privacy + langsmith)
uv sync --all-extras
```

After setup, verify your environment with the doctor:

```bash
# Default profile — checks for analysis commands
uv run self-improve doctor

# Fetch profile — also verifies fetch prerequisites (API key, langsmith extra)
uv run self-improve doctor --profile fetch
```

A green report (exit 0) means the local setup is ready. The doctor is
read-only: it never installs, starts services, or makes network calls.

## Running tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_privacy.py
```

## Running checks

```bash
# Lint
uv run ruff check src tests

# Format check (CI enforces this — run `uv run ruff format src tests` to fix)
uv run ruff format --check src tests

# Type check
uv run pyright src
```

## Before submitting a change

1. Run all tests, linter, format check, and type checker.
2. Use only synthetic data in tests and fixtures. Never include real trace data, credentials, or personal identifiers.
3. Update documentation if the CLI contract changes.
4. Keep changes focused and reviewable.
5. Label heuristic metrics as approximate in their output and docstrings.
6. Never hardcode API keys, endpoints, private project names, usernames, email addresses, or personal paths.
7. Scan your diff for secrets before submitting.

## Privacy rules

- Never commit `.env`, `data/`, or any file under `data/`.
- All trace content is sensitive by default. Anonymization is defense in depth, not a guarantee.
- If you add a new recognizer or pattern, test it with synthetic data only.
- If you add a new source adapter, ensure it goes through the privacy layer before persistence.

## Architecture

See [AGENTS.md](AGENTS.md) for the layered architecture and dependency direction.

Dependency direction: `source -> canonical model -> privacy/storage -> deterministic analysis -> CLI`. No reverse dependency from the core to any specific provider.
