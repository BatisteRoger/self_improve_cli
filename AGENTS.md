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
essential** — the CLI can surface the signals (context jumps, dead
context, compaction events), but whether a local optimization helped or
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
- `privacy/` — anonymization policy, recognizers, placeholder mapping, and sanitization reports.
- `storage/` — safe local artifact layout, atomic writes, metadata, and raw-retention controls.
- `representations/` — deterministic L0/L1/L2/L3 TER builders over canonical sanitized data.
- `metrics/` — tool and context metrics, each labeled with its approximation and assumptions.
- `cli/` — command parsing, stable exit codes, stdout/stderr rules, Markdown/JSON output, and agent-oriented help.

Keep the core representation and metrics layers free of network calls and LLM calls. This preserves recomputability and makes them easy to test.

## CLI usage details

### Fetching from a run ID

LangSmith trace URLs often contain a **run ID** rather than a trace ID. Use
`--from-run` to have the CLI resolve the run ID to its parent trace ID before
fetching:

```bash
self-improve fetch 019ff0e5-7189-713d-8ab9-c032edf9d4dd --from-run
```

The resolution is logged to stderr, and the JSON output includes a
`resolved_from_run` field so you can trace back which run ID was used.

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
- Run the relevant formatter, linter, type checker, and tests before declaring a change complete.
- Keep changes focused and reviewable; do not rewrite unrelated files.
- Label heuristic metrics as approximate in their output and docstrings.
