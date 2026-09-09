# Self-improve CLI

An open-source Python CLI for privacy-preserving analysis of AI-agent execution traces. It turns raw, multi-megabyte traces into compact, progressive representations an analyst agent (or human) can use to understand what happened and identify improvement opportunities.

## Why

Observability tools like LangSmith capture detailed execution traces — every LLM call, tool call, retry, and wrapper. A single trace can be 1–2 MB of JSON across dozens of runs. That's too large to feed into an LLM for analysis, and too noisy for a human to scan by hand.

This CLI compresses raw traces into **token-efficient representations (TER)** at four levels, with anonymization by default. A 1.6 MB trace becomes a ~17 KB narrative — roughly **100–250x smaller** — with the option to drill down to any specific run when you need the full context.

### The improvement triangle

Every observation should be weighed against three axes in tension:

- **Output quality** — is the agent succeeding at its task?
- **Speed** — how long does the agent take?
- **Cost** — how many tokens (and API calls) does the agent consume?

**Agentic engineering is context optimization.** The question is not "how do we make the agent better" in the abstract — it's "where are tokens being spent without contributing to quality, and where would spending more tokens improve the outcome?"

The CLI surfaces the signals (context jumps, stale tool results, compaction events, redundant tool calls). Whether a local optimization helped or harmed the whole task is a judgment that requires stepping back and looking at the entire trace in context — that's where the analyst comes in.

## Who this is for

- **Target Agent** — the agent whose executions are being analyzed. Its traces are the evidence.
- **Analyst Agent** — the agent (or human-assisted workflow) that uses this CLI to inspect traces and form observations.

The CLI is **agent-first**: it is the primary interface for an Analyst Agent. Humans can use it too, but agent ergonomics come first. Every command is bounded, composable, and returns stable references for drill-down.

The CLI produces representations and metrics. It does not propose or apply improvements itself. Interpretation and action stay with the analyst and the human reviewer.

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
self-improve skeleton <trace_id>          # L1: structure and metrics
self-improve narrative <trace_id>         # L2: read the trace story
self-improve run-detail <trace_id> <run_id>  # L3: full content of one run
```

The examples in the rest of this README use the bare `self-improve` form for
brevity; prepend `uv run` when working from the source checkout.

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
self-improve skill-check <trace_id> --expected priorisation-financiere  # Verify trigger (exit 0=match, 1=mismatch)
self-improve skill-check <trace_id> --expected none                     # Verify anti-trigger
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
