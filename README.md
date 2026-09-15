# self-improve-cli

[![CI](https://github.com/BatisteRoger/self_improve_cli/actions/workflows/ci.yml/badge.svg)](https://github.com/BatisteRoger/self_improve_cli/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

**Turn multi-megabyte AI-agent execution traces into compact, anonymized
representations a human — or an analyst agent — can actually read.**

`self-improve` fetches LangSmith traces, anonymizes them before anything
touches disk, and builds token-efficient representations (TER) at four levels
of detail. Built agent-first: every command is bounded, composable, and
returns stable run IDs for drill-down.

## What it looks like

A complex agent execution: **dozens of nested runs, ~2 MB of raw JSON**. The
skeleton view tells the whole story in ~1 KB:

```text
root: agent.pipeline status=success total_tokens=28450 latency=42.12s

  0. [chain] agent.pipeline tokens=28450 latency=42.12s
  1.   [chain] supervisor tokens=28450 latency=41.95s
  2.       [llm] guardrails.input_check latency=0.45s
  3.       [llm] planner tokens=3210 latency=2.15s
  4.       [tool] load_skill latency=0.02s
  5.       [tool] search_subagent tokens=14200 latency=24.80s
  6.             [llm] query_rewriter tokens=1120 latency=1.20s
  7.             [tool] web_search latency=0.85s
  8.             [tool] fetch_document latency=18.40s
  9.             [llm] summarizer tokens=8950 latency=4.10s
 10.       [llm] synthesizer tokens=7040 latency=3.40s
```

The agent took 42 seconds — and now you can see why: it delegated to
`search_subagent` (run 5, ~25 s), where a single `fetch_document` call
(run 8) spent 18 seconds waiting for an external service. Without this view,
all you know is "42 seconds."

Comparing two executions is a clean diff table:

```text
| Metric | Baseline | With Subagent | Delta |
| --- | ---: | ---: | ---: |
| Total tokens | 8,420 | 28,450 | +20,030 |
| LLM calls | 3 | 7 | +4 |
| Latency | 6.2s | 42.1s | +35.9s |

Skills triggered:
- Baseline: (none)
- With Subagent: deep-research, data-synthesis
```

## Why

Observability tools like LangSmith capture detailed execution traces — every
LLM call, tool call, retry, and wrapper. A single trace can be 1–2 MB of JSON
across dozens of runs. That's too large to feed into an LLM for analysis, and
too noisy for a human to scan by hand.

This CLI compresses raw traces into **token-efficient representations (TER)**
at four levels, with anonymization by default. A ~2 MB trace becomes a ~20 KB
narrative or a ~1 KB skeleton — with the option to drill down to any
specific run when you need the full context.

### The improvement triangle

Every observation should be weighed against three axes in tension:

- **Output quality** — is the agent succeeding at its task?
- **Speed** — how long does the agent take?
- **Cost** — how many tokens (and API calls) does the agent consume?

**Agentic engineering is context optimization.** The question is not "how do
we make the agent better" in the abstract — it's "where are tokens being
spent without contributing to quality, and where would spending more tokens
improve the outcome?"

The CLI surfaces the signals (context jumps, stale tool results, compaction
events, redundant tool calls). Whether a local optimization helped or harmed
the whole task is a judgment that requires stepping back and looking at the
entire trace in context — that's where the analyst comes in.

## Who this is for

- **Target Agent** — the agent whose executions are being analyzed. Its
  traces are the evidence.
- **Analyst Agent** — the agent (or human-assisted workflow) that uses this
  CLI to inspect traces and form observations.

The CLI is **agent-first**: it is the primary interface for an Analyst Agent.
Humans can use it too, but agent ergonomics come first. Every command is
bounded, composable, and returns stable references for drill-down.

The CLI produces representations and metrics. It does not propose or apply
improvements itself. Interpretation and action stay with the analyst and the
human reviewer.

## Install

> **Note:** not yet published on PyPI — install from git for now.

```bash
# uv
uv tool install "self-improve-cli[langsmith] @ git+https://github.com/BatisteRoger/self_improve_cli"

# or pipx / pip
pipx install "self-improve-cli[langsmith] @ git+https://github.com/BatisteRoger/self_improve_cli"
```

Extras:

- `langsmith` — the LangSmith trace source, required for `fetch`/`list`
- `privacy` — the Presidio NLP anonymizer (more PII coverage; without it, a
  faster regex-only backend handles secrets and common patterns)

From a dev checkout managed with [uv](https://docs.astral.sh/uv/), prefix
every command with `uv run` — do **not** call the
`.venv\Scripts\self-improve` executable directly.

## Quick start

```bash
self-improve init              # Create .env from template (LangSmith API key)
self-improve list --limit 10   # L0: discover recent traces
self-improve fetch <trace_id>  # Download, anonymize, build all representations
self-improve skeleton <trace_id>          # L1: structure and metrics
self-improve narrative <trace_id>         # L2: read the trace story
self-improve run-detail <trace_id> <run_id>  # L3: full content of one run
```

## Commands

| Level | Command | What you get |
| ----- | ------- | ------------- |
| L0 | `list`, `list-runs`, `list-projects` | Discover which traces exist |
| L1 | `skeleton`, `tool-metrics`, `context-metrics`, `skill-metrics` | Structure, metrics, growth signals |
| L2 | `narrative` | Chronological story with message deltas |
| L3 | `run-detail`, `context-at` | Full untruncated content of one run or one step |
| Nav | `target-timeline`, `error-neighborhood` | Every touch on a target; steps around each error |
| — | `compare` | Diff two traces (tokens, latency, skills) |
| — | `skill-check` | Verify expected skill trigger (exit 0/1) |
| — | `assess` | Attach task & outcome assessment to a trace |
| — | `info` | Trace metadata, assessment, and TER stats |
| — | `prompt pull/list/show/diff` | Manage local copies of LangSmith prompts |
| — | `ati list/show` | Target agent architecture documents |
| — | `doctor` | Local setup diagnostic (read-only) |

### Bounded by default, drill-down on demand

Every command starts compact. Scoping flags bound the output further:

- `skeleton --errors-only` — quick error/cancellation triage
- `narrative --full` — original common-prefix diff (compact per-index diff is the default)
- `target-timeline --compact` — tool name + status per step only
- `run-detail --inputs-only` / `--outputs-only` / `--tool-calls-only` — one section only
- `context-at --inputs-only` / `--outputs-only` / `--tool <id>` / `--from N --to M` — bounded step view

All commands support `--format json` for composable, structured output.

## Skills

The CLI ships with agent skills in the `skills/` directory. Install them into your agent with:

```bash
npx skills add BatisteRoger/self_improve_cli
```

Available skills:

- **navigate-traces** — question-to-view routing: which command answers which question
- **analyze-agent** — systematic trace interpretation with structured findings report
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
self-improve skill-check <trace_id> --expected deep-research  # Verify trigger (exit 0=match, 1=mismatch)
self-improve skill-check <trace_id> --expected none           # Verify anti-trigger
```

`skill-metrics` detects skill invocations and estimates the marginal token and latency cost of each. `compare` shows the delta between two traces. `skill-check` is scriptable: exit 0 means the expected skill was triggered, exit 1 means a mismatch.

## Where to learn more

- [AGENTS.md](AGENTS.md) — project values, architecture, evidence navigation contract, CLI usage details, and working rules.
- [SECURITY.md](SECURITY.md) — threat model, privacy guarantees, and reporting process.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, tests, and review expectations.

## Status

Early release (v0.1.0). LangSmith is the first supported trace source; the architecture is designed for provider interoperability. Current focus: Level 1 — single-trace diagnostic excellence.

## License

[MIT](LICENSE)
