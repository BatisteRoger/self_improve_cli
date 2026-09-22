---
name: improve-agent
description: Run one iteration of the observe-diagnose-intervene-verify loop on a target agent — run its task or eval, fetch the trace, diagnose, persist findings, and produce a compare receipt after a change. Use when improving an agent's quality, cost, or speed across iterations — not for one-off diagnosis (use analyze-agent).
---

# Improve an agent through the loop

One iteration of the improvement loop:

```
run task -> fetch trace -> diagnose -> persist findings ->
candidate intervention -> re-run -> compare -> receipt
```

Each iteration must leave durable artifacts: an assessment, persisted
findings, and a comparison receipt. If the loop produces only conversation,
it produced nothing reusable — the next iteration (or the next analyst)
starts from what was stored, not from your chat.

## The loop

### 1. Run the task

Execute the task or eval on the target agent however the target's
environment provides it (test harness, eval runner, production replay).
Record what was asked — you need it for the assessment. Keep the task
identical across iterations: different tasks are not an experiment.

### 2. Fetch and assess

```bash
self-improve fetch <trace_id>
self-improve assess <trace_id> --task "..." --outcome success|partial|fail|unknown --source human|test|evaluator
```

Assess before judging efficiency — seven searches might be wasteful or
necessary; only the outcome tells you.

### 3. Diagnose

Follow the `analyze-agent` skill: name the question, retrieve evidence via
`navigate-traces`, verify before writing findings. If the target has an ATI
doc, read it first (`self-improve ati show <name>`).

### 4. Persist findings

```bash
self-improve finding add <trace_id> --title "..." --pattern <label> \
  --impact ... --strength ... --axis ... --locus ... \
  --evidence "<run_id>: what it shows" \
  --candidate "..." --validation "how to test it"
```

- Pattern labels and fault loci: `analyze-agent/references/mechanisms.md`.
- Candidate improvements: the `harness.*` vocabulary in the same file.
- A clean trace is not a wasted iteration — record it in the assessment
  notes; it is your baseline.

### 5. Candidate intervention

State the candidate as a single controlled variable: what changes, where,
and what you expect to move (which triangle axis, which metric). More than
one variable per iteration breaks attribution — the compare in step 7
becomes uninformative.

The intervention is applied by whoever owns the target agent. This skill
never edits the target.

### 6. Re-run

Same task, same environment, after the intervention. Fetch the new trace
and assess it the same way.

### 7. Compare and receipt

```bash
self-improve compare <trace_a> <trace_b>
self-improve assess <trace_b>
self-improve skeleton <trace_b> --errors-only
```

The receipt answers two questions:

- **Capability held?** Outcome not regressed, no new errors.
- **Efficiency gained?** Tokens, latency, or call count reduced.

Both yes -> the candidate is worth keeping; say so to the reviewer with the
receipt attached. Capability regressed -> revert or iterate. Either way the
receipts stay on disk for the next iteration.

## Discipline

- **One controlled variable per iteration.** Two changes = no attribution.
- **Same task across iterations.** Otherwise `compare` measures task
  difference, not intervention effect.
- **Held-out honesty.** Do not tune the target on the same tasks you score
  it with. Keep at least some eval tasks out of the diagnosis loop.
- **Findings outlive sessions.** Persist them — cross-trace prevalence is
  what turns a hypothesis into a confirmed mechanism.
- **The loop never applies changes by itself.** A human or the owning dev
  agent decides what ships.
