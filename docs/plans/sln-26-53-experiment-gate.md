# Plan: SLN-26 thin slice + SLN-53 (experiment fetch + gate v2)

Status: plan only, no code. Branch `feat/sln-26-53-experiment-gate`. Written 2026-10-15 against `langsmith==0.12.2`.

## 1. Goal

- `fetch --experiment <name> [--outcome-key KEY|"KEY>=T"]` downloads every run of one LangSmith experiment session, stores each trace with the existing sanitize + TER pipeline, and writes one `experiment.json` receipt that groups trace ids by dataset example.
- `gate --baseline <exp|@file> --candidate <exp|@file>` compares two repeated-run sets and returns a per-example flip verdict above a measured noise floor. Exit 0 pass, 1 fail, 2 inconclusive (with `reason_code`).
- `gate <trace_before> <trace_after>` (v1, SLN-48) keeps its rule and exit codes. It gains a `reason_code` field.

Out of scope: experiment manifests (rest of SLN-26), `--anchor` A/A sets (SLN-56), cache-cost estimation (SLN-54), binarized evaluators (workspace side). No LLM calls, no writes to LangSmith, no re-execution.

## 2. Module layout

| File | Change |
| ---- | ------ |
| `domain/experiment.py` (new) | `OutcomePolicy`, `parse_outcome_key()`, `outcome_from_policy()`, `ExperimentReceipt`, `ExperimentRunRef`, `FetchFailure`. Pure dataclasses + functions. Re-export from `domain/__init__.py`. |
| `domain/records.py` | `autofill_assessment()` gains an optional `policy` (an `OutcomePolicy` or `None`). When set, the outcome comes from `outcome_from_policy()`; when unset, the existing binary rule applies unchanged. Human precedence and fill-empty-only rules unchanged. |
| `sources/__init__.py` | `TraceSource.fetch_experiment(name) -> ExperimentListing`, default raises `NotImplementedError` (same pattern as `fetch_feedback`). `ExperimentListing` is a canonical dataclass in `domain/experiment.py`. |
| `sources/langsmith.py` | `LangSmithSource.fetch_experiment(name)`. SDK objects stop here. |
| `storage/experiments.py` (new mixin) | `experiments_dir = data_root / "experiments"`; `save_experiment`, `load_experiment(session_id)`, `find_experiment(name_or_id)`. Composed into `TraceStore`. |
| `storage/records.py` | `save_feedback(trace_id, scores)` / `load_feedback(trace_id)` -> `data/traces/<trace_id>/feedback.json`. |
| `metrics/gate.py` | v1: add `reason_code` to `decide_verdict` output and `gate_data()`. Add the shared `ReasonCode` StrEnum here. |
| `metrics/gate_sets.py` (new) | gate v2: classification, transition table, noise floor, criteria, confounds, efficiency, `gate_sets_data()`, `build_gate_sets()`. No I/O. |
| `cli/discovery.py` | `fetch`: `trace_id` becomes `nargs="?"`; new `--experiment`, `--outcome-key`. Handler `_cmd_fetch_experiment`. |
| `cli/metrics.py` | `gate`: positionals become `nargs="?"`; new `--baseline`, `--candidate`, `--outcome-key`, `--strict-conditions`. Side loading in a helper `_load_side()`. |

Gate v2 lives in its own module because v1 and v2 share only the reason codes and the efficiency summary; one file per verdict keeps both readable (matches the one-module-per-view convention in `representations/`).

### Storage layout

```text
data/
  experiments/<session_id>/experiment.json   # receipt (new)
  traces/<trace_id>/sanitized.json ...       # unchanged, one dir per experiment trace
  traces/<trace_id>/assessment.json          # outcome propagated from the policy
  traces/<trace_id>/feedback.json            # new: key + numeric score only
```

Traces stay in `data/traces/` so `diff`, `narrative`, `compare`, `assess` work on experiment traces with no change. The receipt only references trace ids. Keyed by session UUID (stable, path-safe through `_validate_id`); names are looked up by scanning receipts.

### `experiment.json` schema (v1)

```json
{
  "schema_version": 1,
  "source": "langsmith",
  "experiment_name": "<session name>",
  "session_id": "<uuid>",
  "dataset_id": "<uuid>",
  "fetched_at": "<ISO8601 UTC>",
  "complete": true,
  "outcome_policy": {"key": "correctness", "threshold": 0.5, "source": "fetch"},
  "conditions": {
    "start_time": "<ISO8601|null>",
    "end_time": "<ISO8601|null>",
    "revision_id": "<str|null>",
    "git_commit": "<str|null>",
    "git_dirty": "<bool|null>",
    "dataset_version": "<ISO8601|null>",
    "dataset_splits": ["base"],
    "models": ["<model id>", "..."]
  },
  "examples": {
    "<example_id>": [
      {"run_id": "<uuid>", "trace_id": "<uuid>", "status": "success", "outcome": "success"}
    ]
  },
  "failures": [
    {"run_id": "<uuid>", "example_id": "<uuid>", "stage": "trace_fetch", "reason": "<short text>"}
  ]
}
```

- `outcome_policy` is `null` when `--outcome-key` was not given. `source` is `fetch` here; gate writes `gate` when it overrides.
- `outcome` per run is `success` / `fail` / `unknown`, derived once at fetch. Gate re-derives from `feedback.json` when the policy is overridden (section 4).
- Example inputs/outputs, feedback comments, and git `remote_url`/`branch` are never persisted (not anonymized; not needed).

## 3. `fetch --experiment`

### Argument rules

- `fetch <trace_id>` XOR `fetch --experiment <name>`. Both or neither -> usage error, exit 2, message names both forms. `--from-run` with `--experiment` -> usage error.
- `--outcome-key` is only valid with `--experiment` (single-trace fetch keeps the binary autofill rule; extending it is not required by either ticket).
- `parse_outcome_key`: `KEY` -> `{key, threshold: null}`; `KEY>=T` -> `{key, threshold: float(T)}`. Only `>=` is accepted. Empty key, non-numeric `T`, or another operator -> `ValueError` with the two valid forms.

### Adapter: `LangSmithSource.fetch_experiment(name)`

1. `_check_project_allowed(name)` — experiment sessions are LangSmith tracing projects, so the allowlist applies to the experiment name. `LANGSMITH_PROJECT` / `--project` play no role: scope comes from the session id.
2. `session = await client.aread_project(project_name=name)`. Missing -> `LookupError` with "run `self-improve list-projects`". `session.reference_dataset_id is None` -> error "not an experiment session (no reference dataset)".
3. Page through `client.datasets.experiment_runs.query(str(dataset_id), experiment_ids=[str(session.id)], page_size=20, selects=[ID, NAME, STATUS, TRACE_ID, PROJECT_ID, REFERENCE_EXAMPLE_ID, FEEDBACK_STATS, TOTAL_TOKENS, PROMPT_TOKEN_DETAILS, EXTRA, METADATA])`. It returns an `AsyncPaginator`; iterate with `async for` inside one `_run_async` call (same pattern as `list_root_runs`). Do not use `runs.query(project_ids=...)`: it returns 0 rows for experiment sessions.
4. For each item: `example_id = item.id`; for each `run` in `item.runs or []`: keep `run.id`, `run.trace_id or run.id`, `run.status`, and `{key: stats.avg for key, stats in (run.feedback_stats or {}).items()}` (plus `n`/`errors` when present, for notes). Drop `item.inputs/outputs/metadata` and `stats.comments` here.
5. Conditions from `session.metadata` (= `session.extra["metadata"]`, filled by the SDK `evaluate()` runner): `revision_id`, `git.commit`, `git.dirty`, `dataset_version`, `dataset_splits`; `session.start_time`, `session.end_time`. Absent keys stay `null` (experiments launched by other SDKs may not record them — unavailable, never guessed).
6. Returns `ExperimentListing(name, session_id, dataset_id, conditions, runs: list[ExperimentRunRef])`.

### Pagination and dedup

- The paginator follows `next_cursor` until exhausted. A safety cap (`_EXPERIMENT_MAX_RUNS = 5000`) stops with a warning and `complete: false` instead of looping forever.
- Dedup by `run_id` (a run seen twice across pages is kept once). Group by `example_id`; within an example, order by run id for deterministic output.
- A run whose `reference_example_id` disagrees with the item's example id: trust `item.id` and log a warning (the endpoint groups by example).

### Per-run fetch (CLI handler)

For each run, in order:

1. If `data/traces/<trace_id>/sanitized.json` exists, skip the download (log "already stored"). Sanitized traces are immutable records; this makes a re-run resume a partial fetch cheaply.
2. Else `source.fetch_trace(trace_id, project_id=session_id)` -> `anonymize_trace` -> `save_sanitized` -> `write_ter`. Same backend resolution and `--keep-raw` semantics as single-trace fetch. Extract the shared steps of `_cmd_fetch` into `_store_trace(source, store, trace_id, project_id, args)` so both paths stay identical.
3. `save_feedback(trace_id, scores)` from the listing (no extra API call; `feedback_stats` avg per key).
4. `autofill_assessment(existing, trace, feedback, policy=policy)` -> save. Human assessments untouched.
5. Record `outcome` in the receipt from `outcome_from_policy(scores, policy)` (or `unknown` without a policy).

Partial failures:

| Case | Behavior |
| ---- | -------- |
| `fetch_trace` raises or returns 0 runs | Append to `failures` (`stage: trace_fetch`), exclude from `examples`, continue. |
| Run has no `feedback_stats`, or the policy key is absent, or `avg is None` (evaluator error / categorical) | Trace stored, `outcome: unknown`, assessment notes say why. Never guessed. |
| Run `status` is `error` | Kept: an agent error is evidence. Outcome still from feedback. |
| Example with 0 stored runs | Kept in `examples` as `[]` so gate can report it as missing. |

`complete` is `false` when `failures` is non-empty or the cap was hit. Exit code: 0 when complete; 1 (`EXIT_ERROR`) when not, with the receipt still written and stderr saying "re-run the same command to resume". The receipt is written atomically at the end (`write_json`); a crash mid-fetch leaves stored traces, which the resume skips.

Output (JSON and Markdown): name, session id, dataset id, examples count, runs per example (min/max), stored/skipped/failed counts, outcome counts (success/fail/unknown), receipt path, and next command `self-improve gate --baseline <name> --candidate <other>`.

### Outcome policy semantics (`outcome_from_policy`)

- No policy -> `unknown` for every run (fetch does not apply the binary autofill rule to experiment runs: experiment evaluators are continuous; the user must declare). Assessment autofill without policy keeps today's binary rule.
- Policy without threshold: score must be exactly 0 or 1 -> `fail` / `success`; anything else -> `unknown` (note: "non-binary score, declare `KEY>=T`").
- Policy with threshold: `score >= T` -> `success`, else `fail`.
- Key missing or score `None` -> `unknown`.
- `avg` is the mean over feedback points for that key on the run; when `n > 1` the note says so. This is recorded as an assumption, not hidden.

## 4. Gate v2

### Resolving a side (`_load_side(arg, store, outcome_override)`)

- `@path` -> id file. Lines: `trace_id` or `example_id trace_id` (whitespace-separated); `#` comments and blank lines ignored. When `example_id` is absent, read `reference_example_id` from the stored root run's `extra` (verified to survive sanitization). Absent there too -> error naming the line and the two valid line forms. Unknown trace id -> error with `self-improve fetch <trace_id>`.
- Anything else -> `store.find_experiment(arg)`: exact session-id dir first, then exact `experiment_name` match across receipts. Not found -> error "fetch it first: `self-improve fetch --experiment '<arg>'`". Gate stays offline.
- Outcome source per side:
  1. `--outcome-key` on gate: applies to both sides, re-derived from each trace's `feedback.json`. Policy recorded with `source: gate`. Traces without `feedback.json` -> `unknown`.
  2. Else experiment side: `experiment.json` `outcome_policy` and the stored per-run `outcome`.
  3. Else `@file` side: each trace's `assessment.json` outcome (`success` -> pass, `fail` -> fail, `partial`/`unknown`/missing -> unknown). Policy recorded as `{"key": null, "threshold": null, "source": "assessment"}`.
- `partial` is never counted as pass or fail in v2 (binary classification needs two states; folding it either way would invent a verdict).

`--baseline` and `--candidate` must both be given and must not be the same resolved set (same session id, or identical trace-id sets) -> error. Mixing them with the v1 positionals -> usage error. Same dataset is checked when both are experiments: differing `dataset_id` -> error (pairing by example id is meaningless). `@file` sides are paired by example id only.

`Side` dataclass (`metrics/gate_sets.py`): `label`, `kind` (`experiment`|`file`), `ref` (session id or path), `policy`, `groups: dict[example_id, list[RunOutcome]]`, `conditions`. Anchor-ready: the noise-floor function takes a `Side`, so SLN-56 can pass an A/A anchor side instead of the baseline without touching the flip arithmetic.

### Per-example classification

Per side, per example, over its k runs with outcome in {pass, fail, unknown}:

| Condition (checked in order) | Class |
| ---- | ----- |
| example absent on this side, or k = 0 | `missing` |
| any run `unknown` | `unknown` |
| k = 1 | `single` |
| all pass | `stable_pass` |
| all fail | `stable_fail` |
| otherwise | `unstable` |

Decision: any unknown run makes the example `unknown` rather than classifying the remaining runs. Evaluator errors may correlate with failures; dropping them would bias toward pass.

### Transition table and per-example status

Examples are the union of both sides' example ids. Per example:

- either side `missing` -> per-example INCONCLUSIVE `quality_unknown` (reported as missing).
- either side `unknown` -> INCONCLUSIVE `quality_unknown`.
- either side `single` -> INCONCLUSIVE `no_noise_estimate` (decision 3; never pairwise).
- else one of the 9 cells of the 3x3 table `{stable_pass, unstable, stable_fail}^2`.

`regression` = `stable_pass -> stable_fail`. `gain` = `stable_fail -> stable_pass`. All other cells are reported but carry no verdict weight (moves into or out of `unstable` are not distinguishable from noise at this k).

### Noise floor and the quality rule (decision 2)

Let `N` = examples that landed in the 3x3 table (comparable). Let `B` = baseline examples classified `stable_pass`, `stable_fail`, or `unstable` (k >= 2, no unknown). `U` = baseline `unstable` count.

- `noise_floor = U / B` (share of self-unstable baseline examples). `B = 0` -> no floor -> quality INCONCLUSIVE `no_noise_estimate`.
- `regression_rate = R / N`, `gain_rate = G / N`.

Quality criterion, in order:

1. `N = 0` -> INCONCLUSIVE, reason `no_noise_estimate` if any example is `single`, else `quality_unknown`.
2. `regression_rate > noise_floor` -> FAIL.
3. `R = 0` -> PASS.
4. else (`0 < regression_rate <= noise_floor`) -> INCONCLUSIVE `within_noise`.

Chosen rule: **regressions alone vs the floor, strictly greater**. Justification:

- The gate is a non-regression gate. Netting `R - G` lets a gain on example B hide a real regression on example A; they are different tasks for different users.
- A `stable_pass -> stable_fail` move happens by chance only when the example is truly unstable and both k-run samples landed at the extremes. The baseline's unstable share is the measured rate of such examples, so it bounds the chance-regression rate from above (approximately; see below).
- Strict `>`: equality means the observation is exactly what noise can produce.

Gains are reported (count, rate, example ids) but do not offset regressions. A quality "improved" status is shown when `R = 0` and `gain_rate > noise_floor`; it does not change the verdict (still PASS).

Honesty notes printed with the result (APPROXIMATE): with small k, an unstable example can show k identical outcomes by chance, so `noise_floor` underestimates true instability; the floor is a per-run observation, not a significance test. Per-example inconclusive examples are listed with their reason.

Worked example (8 examples, k=3): baseline has 1 unstable -> floor = 1/8 = 0.125. Candidate yields R=1, G=0, N=8 -> 0.125 > 0.125 is false, R>0 -> INCONCLUSIVE `within_noise`. R=2 -> 0.25 > 0.125 -> FAIL.

### Outcome policy mismatch (decision 6)

When both sides carry a policy and `(key, threshold)` differ (and no gate `--outcome-key` override):

- confound line `outcome_policy: baseline correctness>=0.5 vs candidate correctness>=0.7`;
- quality criterion INCONCLUSIVE `quality_unknown` for the whole set (not per example: the policy is per side, so every comparison is affected equally). Recovery hint: pass `--outcome-key` to gate to re-derive both sides under one policy from `feedback.json`.

An experiment side with `outcome_policy: null` and no override -> all its runs are `unknown` -> `quality_unknown` with the same hint. An `@file` side using assessments vs an experiment side with a policy -> confound line only (sources differ by construction); quality is computed.

### Efficiency criterion (decision 7)

Per trace: `efficiency_summary(trace.runs)` (tokens, latency_ms, total_tool_calls). Per example and side: median and range (min-max) over its k runs; a metric is `unavailable` for an example when any run's value is `None` (tokens stay `None`, never 0).

Per example and metric (needs both sides available and baseline k >= 2):

- `up` when candidate median > baseline max;
- `down` when candidate median < baseline min;
- `within_spread` otherwise.

Per metric over comparable examples: `degraded` when `#up > #down`, `improved` when `#down > #up`, else `equal`. Plus aggregate median of per-example median deltas and the count of unavailable examples. Efficiency criterion: any metric `degraded` -> INCONCLUSIVE `efficiency_tradeoff`; else PASS. Labeled APPROXIMATE.

### Confounds (decision 5)

Always shown as facts, per side, with `same`/`differs`/`unavailable`:

| Confound | Experiment side | `@file` side |
| -------- | --------------- | ------------ |
| model id | distinct `ls_model_name` values in `run.extra["metadata"]` of significant LLM runs (stored traces) | same |
| commit | `conditions.git_commit` (+ `git_dirty`), fallback `revision_id` | unavailable: no session metadata |
| dataset version | `conditions.dataset_version` (SDK as-of timestamp) and `dataset_splits`; a version tag is not recorded by the SDK -> `version_tag: unavailable` | unavailable |
| wall-clock gap | `candidate.start_time - baseline.end_time` (or start-start when end missing) | min/max root `start_time` of the stored traces (labeled inferred) |
| outcome policy | `outcome_policy` | `assessment` |

`ls_model_name` is the LangChain tracing convention, not a product key; absent -> `unavailable`. The model id may also be computed at fetch and stored in `conditions.models` to avoid reloading traces; gate recomputes for `@file` sides.

`--strict-conditions`: any confound `differs` (unavailable does not count) -> conditions criterion INCONCLUSIVE `conditions_differ`. Without the flag the conditions criterion is informational only. **Flag:** `conditions_differ` is a fifth reason code not in the decided list (see Open questions).

### Verdict (decision 9)

Criteria: `quality`, `efficiency`, and `conditions` (only with `--strict-conditions`). Each is PASS / FAIL / INCONCLUSIVE(reason_code).

- any FAIL -> overall FAIL, exit 1, `reason_code: null`;
- else any INCONCLUSIVE -> overall INCONCLUSIVE, exit 2, `reason_code` = first by priority `quality_unknown`, `no_noise_estimate`, `within_noise`, `conditions_differ`, `efficiency_tradeoff` (most fundamental first); all criterion codes are listed in `criteria`;
- else PASS, exit 0.

Divergence from v1, on purpose: v1 lets "quality improved" pass despite degraded efficiency. v2 is conjunctive per decision 9, so a quality gain with degraded efficiency is INCONCLUSIVE `efficiency_tradeoff`.

### JSON receipt (`gate_sets_data`)

```json
{
  "mode": "sets",
  "baseline": {"label": "...", "kind": "experiment", "ref": "<session_id>", "examples": 8, "runs": 24, "k": {"min": 3, "max": 3}, "outcome_policy": {...}},
  "candidate": {"...": "same shape"},
  "verdict": "inconclusive",
  "reason_code": "within_noise",
  "reason": "1 regression (12.5%) does not exceed the noise floor (12.5%).",
  "rule": "<one-line rule text>",
  "criteria": {
    "quality": {"status": "inconclusive", "reason_code": "within_noise",
      "noise_floor": 0.125, "comparable": 8, "regressions": 1, "gains": 0,
      "regression_rate": 0.125, "gain_rate": 0.0,
      "transitions": {"stable_pass": {"stable_pass": 5, "unstable": 0, "stable_fail": 1}, "unstable": {}, "stable_fail": {}},
      "inconclusive_examples": [{"example_id": "...", "reason_code": "no_noise_estimate"}]},
    "efficiency": {"status": "pass", "approximate": true, "metrics": {"tokens": {"degraded": false, "up": 1, "down": 3, "within_spread": 4, "unavailable": 0, "median_delta": -1200}}},
    "conditions": {"status": "info", "facts": [{"name": "model", "baseline": ["..."], "candidate": ["..."], "state": "same"}]}
  },
  "examples": [{"example_id": "...", "baseline": "stable_pass", "candidate": "stable_fail", "transition": "regression"}],
  "representative_regressions": [
    {"example_id": "...", "baseline_trace": "...", "candidate_trace": "...",
     "next": ["self-improve diff <b> <c>", "self-improve narrative <c> --budget 4000"]}
  ]
}
```

- `representative_regressions`: up to 3 regressions (first by example id). Trace pick: first trace id per side in sorted order (deterministic). Pointers only, no excerpts (decision 8).
- `examples` lists every example in JSON; Markdown shows only non-`stable_pass -> stable_pass` rows, capped at 10 with an omission marker.

### Markdown output (`build_gate_sets`)

```text
# Gate (sets): INCONCLUSIVE (within_noise)
Baseline: <label> (8 examples, k=3)  Candidate: <label> (8 examples, k=3)
Verdict reason: ...
- Quality: inconclusive (within_noise) | noise floor 12.5% (1/8 baseline unstable) | regressions 1/8 | gains 0/8
  3x3 transitions (rows baseline, cols candidate): [table]
- Efficiency (APPROXIMATE): pass | tokens down 3 / up 1 ...
- Conditions: model same | commit differs (abc123 vs def456) | dataset version same | gap 2h10m | policy same
Representative regressions: example_id, before/after trace ids, next commands
Rule: ...
```

### v1 retrofit (decision 4)

`decide_verdict` returns `(verdict, reason_code, reason)`: quality unknown -> `quality_unknown`; stable + efficiency degraded -> `efficiency_tradeoff`; pass/fail -> `None`. `gate_data()` adds `"mode": "pair"` and `"reason_code"`. Markdown header appends the code on inconclusive. Exit codes unchanged. Update `tests/test_gate.py` expectations.

## 5. Test strategy

All synthetic, no network. Follow `tests/helpers.py` / `conftest.py` patterns.

Unit (`tests/test_gate_sets.py`):

- classification: all pass, all fail, mixed, k=1, unknown run, missing side.
- transitions: each of the 9 cells; union of example ids with a side missing.
- noise floor: `U/B` with B counting only k>=2 known examples; B=0 -> `no_noise_estimate`.
- quality rule: R=0 -> pass; R/N == floor -> within_noise; R/N > floor -> fail; floor 0 and R=1 -> fail; gains do not offset regressions; N=0 with singles -> `no_noise_estimate`, without -> `quality_unknown`.
- policy mismatch -> `quality_unknown` + confound line; gate override re-derives from feedback.
- efficiency: median/range, `None` tokens -> unavailable (not 0), up/down/within, degraded -> `efficiency_tradeoff`.
- confounds: same/differs/unavailable; `--strict-conditions` -> `conditions_differ`; unavailable never triggers strict.
- verdict priority ordering of reason codes; exit code mapping.
- representative regressions: pointers only, deterministic choice, cap 3.

Unit (`tests/test_experiment.py`):

- `parse_outcome_key`: `k`, `k>=0.5`, `k>=x` error, `k<=1` error, empty key error.
- `outcome_from_policy`: binary without threshold, non-binary without threshold -> unknown, threshold boundary (`== T` passes), missing key, `None` score.
- `autofill_assessment` with policy; human assessment untouched.
- storage: receipt round-trip, `find_experiment` by id and by name, not found.

Adapter (`tests/test_langsmith_adapter.py`, fake client like existing tests):

- `fetch_experiment` with a fake async paginator over 2 pages: dedup of a repeated run, grouping, comments/inputs dropped, conditions mapped, missing metadata -> nulls, allowlist refusal, session without dataset -> error.

CLI (`tests/test_cli.py` / new `tests/test_cli_gate_sets.py`):

- `fetch --experiment` with a fake source: a run whose trace fetch fails -> receipt `complete: false`, exit 1; resume skips stored traces; `--outcome-key` propagates to assessment.json.
- `gate --baseline/--candidate` with stored receipts and `@file` (both line forms, missing example id error); usage errors (mixing positional and flags, same set twice, different datasets); JSON and Markdown paths; v1 positional path unchanged plus `reason_code`.

Manual live acceptance (Batiste runs; needs `LANGSMITH_API_KEY` and network):

1. `self-improve fetch --experiment "financial-agent baseline (lila-2040, dataset v2 iteration, x3)" --outcome-key "correctness>=<T>"` and the same for `variant A3 (...)`. Expect 8 examples x 3 runs, `complete: true`, outcomes not all unknown.
2. `self-improve gate --baseline "<baseline name>" --candidate "<variant name>" --format json`. Check: transitions sum to N, floor matches a hand count, confounds populated or `unavailable` with reason, exit code matches verdict.
3. Repeat on the `VALIDATION` pair.
4. Swap baseline/candidate: regressions and gains swap; floor changes to the other side's instability (expected).
5. `gate --baseline @ids_a.txt --candidate @ids_b.txt --outcome-key "<same>"` built from the receipts' trace ids: same transitions as step 2.
6. Note in the PR which conditions came back `unavailable` on real data (JS vs Python SDK metadata).

## 6. Sequencing — one PR, five commits

1. `feat(gate): add reason_code to pair gate` — v1 retrofit + `ReasonCode` enum + tests.
2. `feat(domain): outcome policy and experiment receipt` — `domain/experiment.py`, policy autofill, `feedback.json` storage, experiments storage mixin + tests.
3. `feat(fetch): fetch --experiment` — adapter method, `_store_trace` extraction, CLI handler, partial-failure handling + tests.
4. `feat(gate): experiment-set gate with flip verdict` — `metrics/gate_sets.py` + tests.
5. `feat(gate): wire --baseline/--candidate` and docs — CLI, AGENTS.md (gate section, command audit row, fetch section), README command list, `record_choices.md` entries (rule choice, storage layout, unknown-run rule, efficiency rule), grep `skills/` for `gate` usage and update.

Checks before PR: `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run pyright src`, markdownlint on touched `.md`. Then the manual live acceptance above.

## 7. Compliance surface

- New external call: `datasets.experiment_runs.query` (read-only). Its response carries dataset example inputs/outputs: dropped at the adapter boundary, never persisted.
- Feedback comments dropped; only key + numeric score cross the boundary (same rule as `fetch_feedback`).
- Session metadata: only the listed condition fields are persisted; git `remote_url` and `branch` dropped.
- Every trace goes through the existing anonymizer before persistence; `--keep-raw` keeps its opt-in semantics.
- Allowlist applies to experiment names; the API key remains the real access control.
