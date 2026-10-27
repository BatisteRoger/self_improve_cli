# AGENTS.md

Instructions for people and coding agents working on this repository.

## Project purpose

Self-improve CLI is an open-source Python tool that turns raw AI-agent traces into compact, progressive representations an analyst agent or human can use to understand execution behavior and identify improvement opportunities — with privacy by default.

The CLI produces representations and metrics. It does not propose or apply improvements itself. Interpretation and action stay with the analyst and the human reviewer.

## Scope and direction

**Agent-agnostic.** The CLI must work on any traced agent — nothing in the core, commands, or bundled skills may assume a specific company, product, or harness. Domain knowledge about a specific target agent lives in ATI documents (`document-ati` skill, `ati` commands). Organization-specific workflows live outside this repo (e.g. an internal skill library that orchestrates this CLI), never inside it.

**Direction: single-trace diagnostic first, then the improvement loop.** The long-term shape is a loop — run task → fetch trace → diagnose → persist findings → candidate intervention → re-run → compare → receipt. Features that persist analyst state (`assess`, `finding`) exist so the loop has memory across iterations and sessions. The CLI stays read-side throughout: it produces evidence and evaluation receipts; proposing and applying changes stays with the analyst agent and the human reviewer.

## Core concepts

- **Target Agent**: the agent whose executions are being analyzed. Its traces are the evidence. Sometimes called Agent to Improve or ATI (deprecated terms).
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

- `domain/` — canonical trace, run, message, and artifact types; schema versioning and validation. `trace.py` (model), `records.py` (analyst assessments + findings), `experiment.py` (outcome policy + experiment listing).
- `sources/` — a `TraceSource` protocol plus the initial LangSmith adapter. SDK-specific objects stop at this boundary.
- `privacy/` — anonymization policy, recognizers, placeholder mapping, and sanitization reports. Internally split into `patterns` (regex), `presidio_adapter` (NLP), `placeholders` (stable mapping), `report` (summary), `redact` (recursive application + backend merge), and `structural` (mapping-table replacement + key dropping).
- `storage/` — safe local artifact layout, atomic writes, metadata, and raw-retention controls. `base.py` (paths + serialization) plus per-area mixins: `traces`, `records`, `experiments`, `ter`, `prompts`, `ati`; `TraceStore` composes them.
- `representations/` — deterministic L0/L1/L2/L3 TER builders over canonical sanitized data. One module per view (`skeleton`, `narrative`, `run_detail`, `context_at`, `navigation`, `trace_diff`, `tools`) over `common` helpers; `ter.py` writes all derived files. Public names re-exported from `__init__.py`.
- `metrics/` — tool, context, and skill metrics, each labeled with its approximation and assumptions.
- `cli/` — command parsing, stable exit codes, stdout/stderr rules, Markdown/JSON output, and agent-oriented help. `main.py` is a thin entry point; handlers and parser registration live per command family in `common` (shared plumbing), `discovery`, `views`, `metrics`, `records`, `prompts`, `ati`, `system`. Each module exposes `register(sub)`.

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

### Command audit & merge checklist

Per-command contract status (audit table), known gaps, and the merge checklist for new commands live in [`docs/command-audit.md`](docs/command-audit.md). Update the table there when adding or changing an analysis command.

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

`fetch` pre-fills the assessment: `task` from the first human message of the
sanitized root input; `outcome` + `source: evaluator` only when all recorded
feedback scores are binary and agree (all 1 -> success, all 0 -> fail), else
`unknown` with the scores cited in `notes`. It fills empty fields only and
never touches a `source: human` assessment.

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

### Persisted findings

`assess` records the task & outcome; `finding` records the evidence-backed
observations the analyst produces. Findings persist as `findings.json`
alongside the trace — they are the memory that turns single-trace
diagnostics into cross-trace patterns (a pattern seen once is a hypothesis;
seen across traces it is a mechanism).

```bash
self-improve finding add <trace_id> --title "Agent continued after a 403" \
  --pattern tool.ignored_feedback --impact incorrect_result \
  --strength confirmed --axis quality --locus model \
  --evidence "run-9: 403 response" --evidence "run-10: success claimed" \
  --candidate harness.fail_on_auth_error --validation "Re-run the same task"

self-improve finding list <trace_id>       # compact list; --format json for full records
self-improve finding remove <trace_id> f1
self-improve finding clear <trace_id>
```

- `--impact`: `incorrect_result` | `unverified_completion` | `external_side_effect` | `resource_exhaustion` | `no_impact`
- `--strength`: `observed` | `suspected` | `confirmed`
- `--axis`: `quality` | `cost` | `speed`
- `--pattern`, `--secondary-pattern`, `--locus`: free strings — the
  vocabulary in `skills/analyze-agent/references/mechanisms.md` is living
  and deliberately not enforced at the CLI boundary.
- `--evidence` and `--secondary-pattern` are repeatable.
- Optional context fields: `--assessment`, `--next`, `--candidate`,
  `--validation` — they mirror the findings-report format in the
  `analyze-agent` skill.

Finding IDs are sequential (`f1`, `f2`, ...) so reports and later commands
can cite them stably. `info` reports the findings count and IDs.

### Before/after gate

`gate <trace_before> <trace_after>` is the receipt of one improvement-loop iteration: a verdict for a human gate, not a decision.

- **Quality**: the `assess` outcome on each side, ranked success > partial > fail. A missing assessment or an `unknown` outcome on either side makes quality unknown — never guessed.
- **Efficiency** (approximate, single traces): tokens, latency, tool-call count from the `compare` totals (`metrics/efficiency.py`). Any increase counts as degraded; no noise tolerance.
- **Verdict rule** (`metrics/gate.py::decide_verdict`): quality regressed -> `fail`; quality unknown -> `inconclusive`; quality improved -> `pass`; quality stable + efficiency improved/equal -> `pass`; quality stable + efficiency degraded -> `inconclusive` (a cost increase may be deliberate).
- **Exit code mirrors the verdict**: 0 pass, 1 fail, 2 inconclusive. Errors (unknown trace, same ID twice) also exit 1, and argparse usage errors exit 2 — only 0 is unambiguous.

`pass` means "no regression observed on this pair": a stable `fail -> fail` pair with lower cost passes.

The JSON receipt carries `mode: "pair"` and a `reason_code` on inconclusive verdicts (`quality_unknown`, `efficiency_tradeoff`).

### Experiment-set gate

One pair of stochastic runs can only be inconclusive honestly. `gate --baseline <exp|@file> --candidate <exp|@file>` compares two sets of repeated runs over the same dataset examples (`metrics/gate_sets.py`). It is offline: experiments must be fetched first.

```bash
self-improve fetch --experiment "<baseline experiment>" --outcome-key "correctness>=0.5"
self-improve fetch --experiment "<candidate experiment>" --outcome-key "correctness>=0.5"
self-improve gate --baseline "<baseline experiment>" --candidate "<candidate experiment>"
```

- **`fetch --experiment <name>`** lists the session's runs grouped by dataset example, stores each trace through the normal anonymize + TER pipeline, writes `feedback.json` (key, numeric score, and `n` = points averaged; a `note` when a mean of several points is thresholded) per trace, and writes `data/experiments/<session_id>/experiment.json` (example -> runs, outcome policy, run conditions, failures). Example inputs/outputs and feedback comments are dropped at the adapter. `LANGSMITH_ALLOWED_PROJECTS` applies to the experiment name; `LANGSMITH_PROJECT` plays no role. A run whose trace cannot be fetched is listed under `failures`; the command exits 1 with `complete: false`, and re-running it resumes (stored traces are skipped).
- **`--outcome-key KEY` or `"KEY>=T"`** declares which feedback key decides pass/fail. `KEY` needs a binary 0/1 score; `KEY>=T` thresholds a continuous score. A missing key or score leaves the run `unknown`. Without `--outcome-key` every experiment run is `unknown`.
- **Sides**: a stored experiment (name or session ID), or `@file` with one `trace_id` or `example_id trace_id` per line (the example comes from the root run's `reference_example_id` otherwise). `@file` outcomes come from `assess` (`partial` counts as unknown). `gate --outcome-key` re-derives both sides from `feedback.json` under one policy; `feedback.json` exists only for traces stored by `fetch --experiment`, so a trace without it is an error that says so. In an `@file`, a repeated identical line is kept once, and one trace under two examples is a line error.
- **Outcome evidence**: on feedback-derived sides (experiments, and `@file` with `--outcome-key`), a human `assess` outcome (success, partial, fail) replaces the feedback outcome; a human `unknown` does not. Each side reports `outcomes` (`basis`, `human_overrides`, `description`) and `complete` / `failures` from the receipt (`null` when the receipt does not record them).
- **Classification** per example and side: `stable_pass` / `stable_fail` / `unstable` over its k runs; any unknown run -> unknown; k = 1 -> `no_noise_estimate` (never a pairwise fallback). Regression = `stable_pass -> stable_fail`, gain = the reverse; the full 3x3 table is reported.
- **Quality rule**: noise floor = share of `unstable` examples in the baseline. First match: different outcome rules on the two sides (key vs key, or feedback vs `assess`) or an incomplete experiment fetch -> `quality_unknown`; regressions / comparable examples > floor -> `fail`; an example with no run on one side -> `insufficient_coverage`; examples moved from stable to `unstable` / comparable > floor -> `destabilized` (the count is always printed); no regression -> `pass`; otherwise `within_noise`. Gains are reported and never offset regressions.
- **Efficiency** (approximate): per comparable example, the candidate median against the baseline min-max range for tokens, latency, and tool calls; a metric with more examples up than down is degraded -> `efficiency_tradeoff`. Missing tokens are unavailable, never 0; a metric measured on no example is `unavailable`, and when all are, the criterion status is `unavailable` (not a pass).
- **Conditions** are facts (model ids from `ls_model_name`, git commit, dataset version, outcome policy, wall-clock gap): `same` / `differs` / `unavailable`. `--strict-conditions` turns any `differs` into `conditions_differ`.
- **Verdict** is conjunctive: any criterion fails -> `fail` (exit 1); else any inconclusive -> `inconclusive` (exit 2) with the first `reason_code` of `quality_unknown`, `insufficient_coverage`, `no_noise_estimate`, `destabilized`, `within_noise`, `conditions_differ`, `efficiency_tradeoff`; else `pass` (exit 0). Up to 3 representative regressions are given as trace pointers with `diff` / `narrative --budget` next commands.

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
SELF_IMPROVE_ANONYMIZER=regex self-improve fetch <trace_id>
```

The misspelled legacy name `SELFIIMPROVE_ANONYMIZER` is still read, so
existing `.env` files keep working; `SELF_IMPROVE_ANONYMIZER` wins when both
are set. The same rule applies to the two settings below.

Secrets (API keys, JWTs, bearer tokens, private keys, connection strings,
password assignments) are always detected by regex patterns regardless of
backend. The backend only affects PII detection (email, phone, IBAN, credit
card). Regex secret matches take priority over Presidio when they overlap.
The regex PII set also covers French phone numbers, French social security
numbers (NIR), and birth dates in an explicit `ne/nee le <date>` context only,
because a bare date pattern would destroy financial text.

The `anonymizer_backend` field in fetch JSON output records which backend was
used. The `sanitization_report.recognizer_version` field records which detector
was active.

### Structural redaction

Pattern detection cannot know values that only the target agent knows. Two
optional settings, applied by `fetch` before pattern detection, cover that
(`privacy/structural.py`):

- `SELF_IMPROVE_VALUE_MAP="<table_key>:<value_field>:<token_field>"`
  (comma-separated for several tables). Some target agents pseudonymize their
  own data and record the table in the trace: a list of objects under
  `<table_key>`, at any depth. Every exact occurrence of a `<value_field>`
  string anywhere in the trace (root input, runs before the pseudonymization
  step) is replaced by its `<token_field>`, so the target's tag vocabulary
  stays consistent. Longer values are replaced first. `<table_key>` is then
  dropped.
- `SELF_IMPROVE_DROP_KEYS="key_a,key_b"`: every subtree under one of these
  keys, at any depth, becomes a `<dropped:N chars>` marker, so sizes stay
  visible.

A malformed `VALUE_MAP` spec makes `fetch` fail before any network call. The
sanitization report counts replacements as `MAPPED_VALUE` and dropped
subtrees as `DROPPED_KEY`.

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
- CI (`.github/workflows/ci.yml`) runs the same four commands on ubuntu+windows × Python 3.12/3.13 after `uv sync --extra langsmith --extra dev`. To reproduce CI locally, sync the same extras first. A test that fails locally will fail CI — never dismiss a failure as "environment" without confirming it on a clean `uv sync`.
- Keep changes focused and reviewable; do not rewrite unrelated files.
- Label heuristic metrics as approximate in their output and docstrings.
- Use as few special characters as practical in authored CLI output and documentation. Special characters may still arrive through user conversations or trace content and must be handled robustly — never crash on an unencodable code point.
