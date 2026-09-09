# AGENTS.md

Instructions for people and coding agents working on this repository.

## Project purpose

Self-improve CLI is an open-source Python tool that turns raw AI-agent traces into compact, progressive representations an analyst agent or human can use to understand execution behavior and identify improvement opportunities — with privacy by default.

The CLI produces representations and metrics. It does not propose or apply improvements itself. Interpretation and action stay with the analyst and the human reviewer.

## Core concepts

- **Target Agent**: the agent whose executions are being analyzed. Its traces are the evidence.
- **Analyst Agent**: the agent, or human-assisted workflow, that uses this CLI to inspect traces and form observations.
- **Trace**: one end-to-end operation, composed of runs.
- **Run**: one unit of work, such as an LLM call or tool call.
- **TER**: token-efficient representation; a condensed, recomputable view of trace data.

## The improvement triangle

Every observation and suggestion should be weighed against three axes:

- **Output quality** — is the agent succeeding at its task?
- **Speed** — how long does the agent take?
- **Cost** — how many tokens (and API calls) does the agent consume?

These axes are in tension. The choice of LLM impacts all three, but
context optimization — where tokens are spent and where they are saved —
can improve speed and cost while hurting quality, or while improving it
too. Reducing tokens on the supervisor prompt, trimming a subagent's
tool outputs, or compacting conversation history are all context
optimizations with different trade-offs.

**Agentic engineering is context optimization.** The question is not
"how do we make the agent better" in the abstract — it's "where are
tokens being spent without contributing to quality, and where would
spending more tokens improve the outcome?"

The nuance: sometimes we save tokens in the short term (one step, one
tool call) but harm the conversation overall. A tool output that was
trimmed might contain the information the agent needed three steps
later. A compaction that dropped a constraint might cause a goal drift
that only surfaces at the end. **This is where human investigation is
essential** — the CLI can surface the signals (context jumps, stale
tool results, compaction events), but whether a local optimization helped or
harmed the whole task is a judgment that requires stepping back and
looking at the entire trace in context.

## Values

1. **Token efficiency.** Every output is bounded and progressive. Start with cheap overviews, drill down only when needed.
2. **Designed for agents first.** The CLI is the primary interface for an Analyst Agent. Humans can use it too, but agent ergonomics come first.
3. **Privacy by default.** Traces are anonymized on ingestion. Raw retention is explicit and local-only. Anonymization is defense in depth, not a guarantee.
4. **Deterministic and inspectable layers.** Representations and metrics are pure functions over sanitized data — recomputable, testable, traceable.
5. **Clean boundaries and interoperability.** Provider-specific code stops at a source adapter. The core works on a canonical trace model.
6. **Simple.** Minimum code that solves the problem, no speculative abstractions.
7. **Human in the loop.** Suggestions are produced for review. A human decides what is merged or shipped.

## Non-negotiable principles

1. Keep raw traces as the source of truth for analysis. Every summary, metric, and observation must be traceable back to source runs.
2. Keep the Analyst Agent and Target Agent conceptually separate: the Analyst Agent observes the Target Agent; it must not silently modify it.
3. Separate proposing improvements from evaluating them. A change should be judged by a separate evaluator or a human reviewer.
4. Evaluate executions in context: document what the Target Agent is supposed to do before interpreting a trace.
5. Distinguish a poor overall trajectory from a failed individual run.
6. Prefer small, low-risk, reviewable changes. Do not silently modify the Target Agent or its model.
7. Never invent facts about a trace, agent, or integration; verify them or state the uncertainty.
8. Anonymize before persisting. Never write raw trace content to disk unless an explicit opt-in is given.

## Architecture layers

Dependency direction: `source -> canonical model -> privacy/storage -> deterministic analysis -> CLI`. No reverse dependency from the core to any specific provider.

- `domain/` — canonical trace, run, message, and artifact types; schema versioning and validation.
- `sources/` — a `TraceSource` protocol plus the initial LangSmith adapter. SDK-specific objects stop at this boundary.
- `privacy/` — anonymization policy, recognizers, placeholder mapping, and sanitization reports. Internally split into `patterns` (regex), `presidio_adapter` (NLP), `placeholders` (stable mapping), `report` (summary), and `redact` (recursive application + backend merge).
- `storage/` — safe local artifact layout, atomic writes, metadata, and raw-retention controls.
- `representations/` — deterministic L0/L1/L2/L3 TER builders over canonical sanitized data.
- `metrics/` — tool, context, and skill metrics, each labeled with its approximation and assumptions.
- `cli/` — command parsing, stable exit codes, stdout/stderr rules, Markdown/JSON output, and agent-oriented help.

Keep the core representation and metrics layers free of network calls and LLM calls. This preserves recomputability and makes them easy to test.

## Evidence navigation contract

Every analysis command (existing and future) should satisfy this contract.
The CLI and skills form one interface for the Analyst Agent — the contract
governs both.

1. **Predictable** — consistent identifiers, selectors, output structure, and
   terminology across commands. An analyst should not need to learn different
   step-number conventions across views. Command names describe the question
   being answered, not the internal module name.

2. **Bounded** — compact defaults; selective expansion; explicit omissions.
   A way to continue: "show me what was omitted" without re-running the whole
   command. Never silently turn "show me the relevant evidence" into a huge
   context dump. Distinguish CLI output bounds (enforceable by the CLI) from
   analyst token budget (requires cooperation with the harness).

3. **Connected** — outputs expose stable references to related evidence:
   parent run and child runs, tool call → tool result pairing, tool result →
   subsequent model input containing it, successive model calls in the same
   execution branch. An analyst should be able to follow a reference from one
   view to another without guessing.

4. **Honest** — clearly distinguish recorded facts, approximate measurements
   (labeled as approximate), inferred relationships (labeled as inferred), and
   unavailable evidence (labeled as unavailable). "Not recorded" must not look
   like "did not happen." Repeated action ≠ unnecessary action. Recorded tool
   error ≠ feedback the model received. Missing verification ≠ incorrect
   result.

5. **Recoverable** — an invalid selector or missing artifact produces a
   precise explanation and a valid next action. Recovery should not require
   guessing flags or inspecting Python source. Error messages suggest the
   correct command or selector.

6. **Composable** — readable Markdown for direct consumption; genuinely
   structured JSON where programmatic selection is useful. Avoid making agents
   parse prose to recover identifiers already known to the CLI. Useful next
   actions should be grounded in available data — not generated diagnoses.

### Command audit (as of Wave 5)

| Command | Predictable | Bounded | Connected | Honest | Recoverable | Composable (JSON) |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| `skeleton` | ✓ | ✓ `--errors-only` | run IDs | ✓ | ✓ | ✓ structured |
| `narrative` | ✓ | ✓ truncation, `--from/--to/--around-step` | run IDs per step | ✓ | n/a | ✓ structured |
| `run-detail` | ✓ | ✓ `--tool-calls-only`, `--inputs-only/--outputs-only` (mutually exclusive) | parent/child refs | ✓ | ✓ suggests skeleton | ✓ structured |
| `context-at` | ✓ step index | ✓ bounded preview, `--inputs-only/--outputs-only` (mutually exclusive), `--tool <id>`, `--full` | prev/next step, run-detail | ✓ inferred labels | ✓ lists valid steps, enforces `--from/--to` pairing | ✓ structured |
| `target-timeline` | ✓ | ✓ truncation markers, `--compact` | run IDs per touch | ✓ | ✓ reports no matches | ✓ structured |
| `error-neighborhood` | ✓ | ✓ window-bounded | run IDs, next step | ✓ excludes infra-cancelled | n/a | ✓ structured |
| `tool-metrics` | ✓ | ✓ | n/a (aggregate) | ✓ approximate labels | n/a | ✓ structured |
| `context-metrics` | ✓ | ✓ | step indices | ✓ approximate labels | n/a | ✓ structured |
| `skill-metrics` | ✓ | ✓ | step indices | ✓ approximate labels | n/a | ✓ structured |
| `tools` | ✓ | ✓ `--detail` | tool names | ✓ | n/a | markdown-wrapped (tabular) |
| `compare` | ✓ | ✓ | trace IDs | ✓ | n/a | ✓ structured |
| `skill-check` | ✓ | ✓ | n/a | ✓ | ✓ suggests skill-metrics | ✓ structured |
| `assess` | ✓ | ✓ | n/a | ✓ | ✓ suggests `--task` | JSON in both formats |
| `info` | ✓ | ✓ | n/a | ✓ | ✓ | JSON in both formats |

Gaps (deferred or low-impact):

- `tools` (overview/detail) still returns markdown-wrapped JSON. Low
  priority — the matrix is inherently tabular.
- `assess` and `info` output pretty-printed JSON in both `--format markdown`
  and `--format json` modes (no human-readable markdown representation).
  By design — these are structured metadata, not narrative content.

### Merge checklist for new commands

Before merging a new analysis command, verify:

- [ ] **Predictable**: the command name describes the question it answers;
  selectors (trace_id, run_id, step) use the same conventions as existing
  commands.
- [ ] **Bounded**: default output is compact; expansion is opt-in (`--full`,
  `--from/--to`, `--tool`); truncation uses the shared `_truncate()` marker
  `…[truncated, N chars total]`.
- [ ] **Connected**: output exposes stable run IDs and references to related
  views (parent/child, prev/next step, or the originating tool call).
- [ ] **Honest**: approximate metrics are labeled "APPROXIMATE"; inferred
  relationships are labeled "inferred"; unavailable evidence is stated
  explicitly, not silently omitted.
- [ ] **Recoverable**: invalid selectors produce a precise error and a valid
  next command (e.g. "run `self-improve skeleton <trace_id>` for valid IDs").
- [ ] **Composable**: `--format json` returns a structured dict (not
  `{"content": "<markdown>"}`) when programmatic selection is useful; add a
  `*_data()` function in the representation/metrics layer and wire the CLI
  to use it.
- [ ] **Tests**: focused tests for the new behavior (markdown + json paths,
  error recovery, empty-trace edge case).
- [ ] **Docs**: AGENTS.md command audit table updated; skill files updated if
  the command changes the navigation workflow.

## CLI usage details

### Fetching from a run ID

LangSmith trace URLs often contain a **run ID** rather than a trace ID. Use
`--from-run` to have the CLI resolve the run ID to its parent trace ID before
fetching:

```bash
self-improve fetch 019ff0e5-7189-713d-8ab9-c032edf9d4dd --from-run
```

The resolution tries the **configured project first** (SmithDB-native
`runs.retrieve`). If the run is not in the configured project, it falls back
to a legacy global lookup (`read_run`, deprecated, removed 31 Jan 2027). To
resolve a run from a different project without relying on the deprecated
fallback, specify the project explicitly:

```bash
self-improve fetch <run_id> --from-run --project <project_name>
```

The resolution is logged to stderr, and the JSON output includes a
`resolved_from_run` field so you can trace back which run ID was used.

### Task & outcome assessment

Before judging efficiency, the analyst needs to establish what the agent was
asked to do and whether it succeeded. Use `assess` to attach a manual
assessment to a stored trace:

```bash
self-improve assess <trace_id> --task "Fix the login bug" --outcome success --source human --notes "All tests pass."
```

- `--outcome`: `success` | `partial` | `fail` | `unknown`
- `--source`: `human` | `test` | `evaluator` | `unknown`
- `--notes`: free-form notes about what was produced, verified, or remains unknown

Show the current assessment:

```bash
self-improve assess <trace_id>
```

Clear it:

```bash
self-improve assess <trace_id> --clear
```

When an assessment exists, `skeleton` shows it as a header line and `info`
includes it in the JSON output. This gives the analyst outcome context
before interpreting metrics — seven searches might be wasteful or necessary
depending on the task and its outcome.

"No verification is visible in this trace" is different from "the change is
incorrect." Keep unknown outcomes explicitly unknown.

### Anonymizer backend

The `fetch` command anonymizes traces before persistence. Two backends are
available:

| Backend | `recognizer_version` | When to use |
|---------|---------------------|-------------|
| `auto` (default) | `presidio+regex` or `regex-fallback` | General use — best available |
| `presidio` | `presidio+regex` | Force Presidio; errors if unavailable |
| `regex` | `regex-only` | Faster, no NLP models. Secrets fully detected; PII less comprehensive |

Control via CLI flag or env var (CLI flag overrides env var):

```bash
self-improve fetch <trace_id> --anonymizer regex
SELFIIMPROVE_ANONYMIZER=regex self-improve fetch <trace_id>
```

Secrets (API keys, JWTs, bearer tokens, private keys, connection strings,
password assignments) are always detected by regex patterns regardless of
backend. The backend only affects PII detection (email, phone, IBAN, credit
card). Regex secret matches take priority over Presidio when they overlap.

The `anonymizer_backend` field in fetch JSON output records which backend was
used. The `sanitization_report.recognizer_version` field records which detector
was active.

### Project and workspace configuration

The CLI has three layers of project/workspace control. Only one is a security
control; the other two are convenience:

| Layer | Purpose | Type |
|-------|---------|------|
| `LANGSMITH_API_KEY` | Real access control | Security — enforced server-side by LangSmith |
| `LANGSMITH_PROJECT` (.env) | Default project for `list`/`fetch` | Convenience — override with `--project` |
| `LANGSMITH_ALLOWED_PROJECTS` (.env) | Optional typo-prevention allowlist | Safety net — not a security boundary |

`--project` overrides `LANGSMITH_PROJECT` per-command. The allowlist (if set)
still applies to both. Neither `.env` setting prevents access — the API key's
workspace scoping does.

For `prompt pull`, use `--workspace <UUID>` to pull from non-default LangSmith
workspaces (e.g. Flows). The workspace ID is a LangSmith UUID, not a name.

### Skill analysis commands

When the target agent uses skills (loaded via `read_file` on `SKILL.md` files),
these commands measure trigger accuracy and cost:

- `skill-metrics <trace_id>` — detects skill invocations and estimates marginal token/latency cost.
- `compare <trace_a> <trace_b>` — diffs two traces (tokens, latency, tool calls, skills). Useful for A/B testing skills on vs off.
- `skill-check <trace_id> --expected <skill_name|none>` — scriptable trigger verification. Exit 0 = match, exit 1 = mismatch (false positive or wrong skill).

Skill detection is heuristic: a tool call (`read_file` or `grep`) whose target
path contains `SKILL.md` is counted as a skill invocation. Token cost is
approximate — it assumes the skill is the only cause of context growth between
two LLM steps. Cross-reference with `tool-metrics` and `context-metrics`.

### Doctor: local setup diagnostic

```bash
self-improve doctor                    # default profile
self-improve doctor --profile fetch    # escalate fetch prerequisites to fail
self-improve doctor --format json      # machine-readable output
```

Read-only local setup diagnostic. No installs, no network, no mutations.
A green report (exit 0) means the local environment is consistent enough to
build, run, and test the CLI. Exit 1 = one or more failed checks.

Profiles:

- `default`: checks needed to run analysis commands (skeleton, narrative,
  metrics) on already-fetched traces.
- `fetch`: escalates `langsmith_extra` and `env_api_key` to `fail`, since the
  `fetch` command cannot work without them.

What is **not** checked: API key validity (would require a network call),
dependency freshness (would require an install), artifact freshness, Docker,
databases. See `cli/doctor.py` for the full check list and scope honesty notes.

## Security and open-source hygiene

- Treat all trace content as sensitive. Do not commit raw traces, sanitized local data, credentials, customer content, or private source code.
- Read the repository ignore rules before adding data or fixtures. Use synthetic, minimized examples for tests and documentation.
- Never hardcode API keys, endpoints containing credentials, private project names, usernames, email addresses, or personal paths.
- Configuration must come from documented environment variables or explicit local configuration. Keep secrets out of logs and error messages.
- Before publishing, scan the complete diff and repository history for secrets and personal identifiers.
- The project is licensed under MIT (see LICENSE). Do not add contributor attribution or ownership claims beyond what the LICENSE file already states.

## Working rules

- Follow the existing Python tooling and style; inspect nearby code before introducing abstractions or dependencies.
- Keep generated artifacts out of version control and make derived outputs reproducible.
- Add or update tests for behavior changes, using synthetic data only.
- Update user-facing documentation when the CLI contract changes.
- Run the relevant formatter, linter, type checker, and tests before declaring a change complete. Specifically: `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, and `uv run pyright src`.
- Keep changes focused and reviewable; do not rewrite unrelated files.
- Label heuristic metrics as approximate in their output and docstrings.
- Use as few special characters as practical in authored CLI output and documentation. Special characters may still arrive through user conversations or trace content and must be handled robustly — never crash on an unencodable code point.
