# Self-improve CLI

A small, open-source CLI that turns raw AI-agent traces into compact, progressive representations an analyst agent (or human) can actually use.

## The problem

Observability tools like LangSmith capture detailed execution traces — every LLM call, tool call, retry, and middleware wrapper. A single trace can be 1–2 MB of JSON across dozens of runs. That's too large to feed into an LLM for analysis, and too noisy for a human to scan by hand.

This CLI compresses raw traces into **token-efficient representations (TER)** at four levels, with anonymization by default.

## Commands

| Level | Command | What you get |
| ----- | ------- | ------------- |
| L0 | `list`, `list-runs`, `list-projects` | Discover which traces exist |
| L1 | `skeleton`, `tool-metrics`, `context-metrics`, `skill-metrics` | Structure, metrics, growth signals |
| L2 | `narrative` | Chronological story with message deltas (~15 KB) |
| L3 | `run-detail` | Full untruncated content of one run |
| — | `compare` | Diff two traces (tokens, latency, skills) |
| — | `skill-check` | Verify expected skill trigger (exit 0/1) |
| — | `prompt pull/list/show/diff` | Manage local copies of LangSmith prompts |
| — | `ati list/show` | Target agent architecture documents |

A 1.6 MB trace becomes a ~17 KB narrative — roughly **100–250x smaller** — with the option to drill down to any specific run when you need the full context.

## Quick start

```bash
pip install self-improve-cli[langsmith]
```

Or, from a dev checkout managed with [uv](https://docs.astral.sh/uv/), prefix
every command with `uv run` — do **not** call the `.venv\Scripts\self-improve`
executable directly:

```bash
self-improve init              # Create .env from template
self-improve list --limit 10   # L0: discover recent traces
self-improve fetch <trace_id>  # Download, anonymize, build all representations
self-improve narrative <trace_id>   # L2: read the trace story
```

The examples in the rest of this README use the bare `self-improve` form for
brevity; prepend `uv run` when working from the source checkout.

## Skills

The CLI ships with agent skills in the `skills/` directory. Install them into your agent with:

```bash
npx skills add BatisteRoger/self_improve_cli
```

Available skills:

- **navigate-traces** — the top-down trace analysis workflow (L0 to L3)
- **analyze-agent** — systematic trace interpretation with structured observation templates
- **document-ati** — create architecture documents for target agents being analyzed

```bash
self-improve skill                 # list available skills
self-improve skill navigate-traces # print a specific skill
```

## Skill analysis

When your target agent uses skills (loaded via `read_file` on `SKILL.md` files), these commands help measure trigger accuracy and cost:

```bash
self-improve skill-metrics <trace_id>     # Which skills were triggered, token cost
self-improve compare <trace_a> <trace_b>  # Diff two traces (e.g. skills on vs off)
self-improve skill-check <trace_id> --expected priorisation-financiere  # Verify trigger (exit 0=match, 1=mismatch)
self-improve skill-check <trace_id> --expected none                     # Verify anti-trigger
```

`skill-metrics` detects skill invocations and estimates the marginal token and latency cost of each. `compare` shows the delta between two traces. `skill-check` is scriptable: exit 0 means the expected skill was triggered, exit 1 means a mismatch.

## Where to learn more

- [AGENTS.md](AGENTS.md) — project values, architecture, principles, CLI usage details, and working rules.
- [SECURITY.md](SECURITY.md) — threat model, privacy guarantees, and reporting process.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, tests, and review expectations.

## Status

Early release (v0.1.0). LangSmith is the first supported trace source; the architecture is designed for provider interoperability.

## License

[MIT](LICENSE)
