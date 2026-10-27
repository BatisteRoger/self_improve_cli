# Command Audit — Evidence Navigation Contract

Per-command status against the six contract criteria defined in `AGENTS.md` § Evidence navigation contract, plus the merge checklist for new commands. Update the table when adding or changing an analysis command.

## Audit (as of Wave 5)

| Command | Predictable | Bounded | Connected | Honest | Recoverable | Composable (JSON) |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| `skeleton` | ✓ | ✓ `--errors-only` | run IDs | ✓ | ✓ | ✓ structured |
| `narrative` | ✓ | ✓ truncation, `--from/--to/--around-step`, `--budget` (header and omission markers counted; Markdown and JSON keep the same steps) | run IDs per step + omitted markers | ✓ | ✓ validates `--budget > 0` | ✓ structured |
| `run-detail` | ✓ | ✓ `--tool-calls-only`, `--inputs-only/--outputs-only` (mutually exclusive) | parent/child refs | ✓ | ✓ suggests skeleton | ✓ structured |
| `context-at` | ✓ step index | ✓ bounded preview, `--inputs-only/--outputs-only` (mutually exclusive), `--tool <id>`, `--full` | prev/next step, run-detail | ✓ inferred labels | ✓ lists valid steps, enforces `--from/--to` pairing | ✓ structured |
| `target-timeline` | ✓ | ✓ truncation markers, `--compact` | run IDs per touch | ✓ | ✓ reports no matches | ✓ structured |
| `error-neighborhood` | ✓ | ✓ window-bounded | run IDs, next step | ✓ excludes infra-cancelled | n/a | ✓ structured |
| `diff` | ✓ narrative step numbers (`A#N = B#M`) | ✓ unchanged runs collapsed, divergence blocks capped at 10 steps per side with `narrative --from/--to` markers, `--full` | run IDs on both sides | ✓ alignment by `(run_type, name)` stated; no scoring | ✓ same ID, empty trace, unknown trace | ✓ structured |
| `tool-metrics` | ✓ | ✓ friction section only when retries/oversized outputs exist; oscillations and pattern labels JSON-only | run IDs on retry/oversized signals | ✓ approximate + merged parallel latency; retries need same target within 5 calls | n/a | ✓ structured |
| `context-metrics` | ✓ | ✓ lifecycle section only above re-paid threshold or with duplicates, top 3 non-system items + top 3 duplicates; churn JSON-only | step indices + `tool_call_id`s | ✓ approximate labels; drops/churn scoped to the same node (inferred) | n/a | ✓ structured |
| `skill-metrics` | ✓ | ✓ | step indices | ✓ approximate labels | n/a | ✓ structured |
| `tools` | ✓ | ✓ `--detail` | tool names | ✓ | n/a | markdown-wrapped (tabular) |
| `compare` | ✓ | ✓ | trace IDs | ✓ partial token usage shown as a labeled lower bound | n/a | ✓ structured |
| `gate` | ✓ before/after order | ✓ verdict + per-axis reasons | `diff`/`compare`/`assess` next commands | ✓ missing/unknown outcome = inconclusive; efficiency labeled approximate | ✓ same ID, unknown trace | ✓ structured; exit 0/1/2 mirrors verdict; `reason_code` |
| `gate --baseline/--candidate` | ✓ baseline/candidate order, same side forms | ✓ 3x3 table, flagged examples capped at 10 in Markdown, full list in JSON | example IDs + trace pointers with `diff`/`narrative` next commands | ✓ noise floor and efficiency labeled approximate; unknown/k=1 never judged; conditions `unavailable` stated | ✓ unknown experiment suggests `fetch --experiment`, id-file line errors, usage errors | ✓ structured; `reason_code` |
| `skill-check` | ✓ | ✓ | n/a | ✓ | ✓ suggests skill-metrics | ✓ structured |
| `assess` | ✓ | ✓ | n/a | ✓ | ✓ suggests `--task` | JSON in both formats |
| `finding` | ✓ subcommands like `ati`/`prompt` | ✓ compact list, JSON for full | finding IDs, run IDs in evidence | ✓ free-string pattern/locus documented | ✓ suggests `finding list` | ✓ structured |
| `info` | ✓ | ✓ | n/a | ✓ | ✓ | JSON in both formats |

Gaps (deferred or low-impact):

- `tools` (overview/detail) still returns markdown-wrapped JSON. Low
  priority — the matrix is inherently tabular.
- `assess`, `info`, and `finding add` output pretty-printed JSON in both
  `--format markdown` and `--format json` modes (no human-readable
  markdown representation). By design — these are structured metadata,
  not narrative content.

## Merge checklist for new commands

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
- [ ] **Docs**: this audit table updated; skill files updated if the command
  changes the navigation workflow.
