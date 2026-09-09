---
name: analyze-agent
description: Focused single-trace diagnostic — take a trace and an engineering question, retrieve the relevant evidence, produce a supported findings report or explain why you cannot. Use when diagnosing an AI agent's execution trace to identify improvement opportunities.
---

# Diagnose an agent trace

A focused workflow for answering an engineering question about a single trace.
Produces a small, evidence-backed findings report — or explicitly states why
no supported finding can be made. Interpretation stays with the analyst; the
skill guides evidence retrieval, not judgment.

## Prerequisites

```bash
self-improve fetch <trace_id>   # Download, anonymize, and build representations
```

If the target agent has an architecture document, read it first:

```bash
self-improve ati show <agent_name>
```

Architecture context turns "the agent called search 7 times" into "the agent's
retrieval strategy is inefficient because it doesn't consolidate results."

## Step 0: Check the assessment and sanitization

```bash
self-improve assess <trace_id>
self-improve info <trace_id> --format json
```

The assessment tells you what the agent was asked to do and whether it
succeeded. This context changes how you interpret everything else — seven
searches might be wasteful or necessary depending on the task and its
outcome. The skeleton also shows the assessment as a header line when set.

Check the sanitization report from `info`: if `complete: false`, anonymization
may be incomplete — be careful about quoting trace content verbatim.

If no assessment exists, proceed without it — but consider setting one
before drawing conclusions about efficiency.

## Step 1: Name the question

Before running any command, state the question you are trying to answer.
"Why did it fail?" is too broad. "Why did the `edit_file` call at step 7
fail, and did the model receive the error?" is precise enough to guide
evidence retrieval.

Common question types:

- "Why did this tool call fail?"
- "Did the model receive that tool result/error?"
- "Why did it read this file 4 times?"
- "Did compaction drop the user's constraint?"
- "What supports the final claim?"
- "Where did this become expensive?"
- "Is this trace clean?"

## Step 2: Retrieve evidence (question-driven)

Use the `navigate-traces` skill to pick the cheapest view that can answer
your question. Do not run every command — run the one that answers the
question, then drill down only if insufficient.

```bash
self-improve skeleton <trace_id>          # Always: the shape of the trace
```

Then, depending on the question:

| Question | Next command | Drill-down if insufficient |
|----------|-------------|---------------------------|
| "Which steps errored or were cancelled?" | `skeleton --errors-only` | `error-neighborhood` for context around each error |
| "Why did this tool call fail?" | `run-detail` on the tool run | `context-at` for the next model input |
| "Did the model receive that?" | `context-at` for the step | `run-detail --inputs-only` on the next LLM run |
| "Why did it read this file 4 times?" | `tool-metrics` | `target-timeline --compact` (overview), then `run-detail` on each touch |
| "Did compaction drop the constraint?" | `context-metrics` (look for a drop) | `context-at --from N --to M` across the drop |
| "What supports the final claim?" | `narrative` (find the claim) | `run-detail --outputs-only` on the last LLM run |
| "Where did this become expensive?" | `context-metrics` growth curve | `tool-metrics` for attribution |
| "Is this trace clean?" | `skeleton --errors-only` (quick triage) | `tool-metrics`, `context-metrics`, `error-neighborhood` |
| "Did the agent recover from that error?" | `error-neighborhood` | `run-detail` on the error and the next step |

Read each command's output before deciding whether to drill down. The
skeleton is always worth running first — it shows the shape of the trace
and identifies runs worth investigating.

## Step 3: Verify the evidence

Before writing a finding, verify that the evidence actually supports it:

- A recorded tool error does not guarantee the model received it as
  feedback. Use `context-at` to check the next model input.
- Repeated calls on the same target may be legitimate if the target
  changed between calls. Use `target-timeline` to verify.
- A metric signal (context jump, stale ratio) is a fact, not a conclusion.
  It points to where to look, not to what went wrong.
- "No visible verification" is different from "the result is incorrect."
  State which one the evidence supports.

## Stopping rules

Stop investigating when any of these is true:

1. **The question is answered.** You have evidence-backed findings with
   citations to specific run IDs.
2. **Evidence is unavailable.** The trace does not contain the information
   needed (e.g. compaction removed the context, or the run was not
   recorded). State this explicitly — "evidence unavailable" is a valid
   conclusion.
3. **Further inspection won't change the conclusion.** You've checked the
   plausible alternatives and they don't hold. Don't keep digging for
   completeness.
4. **Budget is exhausted.** Summarize what you found and what remains
   unverified.

Do not force a full tour of every view when the question is already
answered. Do not invent findings, alternative explanations, or extra tool
calls to fill a template.

## Step 4: Write the findings report

Produce a small number of findings (typically 1–5), ranked by consequence
and evidence strength. Not every trace has a problem — "no supported
finding" is a valid and honest outcome.

### Finding format

```markdown
### Finding: <concise description>

**Pattern**: <canonical label from mechanisms.md>
**Impact**: <incorrect_result | unverified_completion | external_side_effect | resource_exhaustion | no_impact>
**Evidence strength**: <observed | suspected | confirmed>
**Triangle axis**: <quality | cost | speed> — <how affected>

**Evidence**:
- <run_id>: <what the trace shows — quote or metric value>
- <run_id>: <additional evidence>

**Assessment**:
<what the evidence establishes, and what it does NOT establish>

**Likely locus**: <model | agent_harness | context | memory | tool_integration | external_environment | task_spec>

**Suggested next action**: <what to verify next, not a fix>

**Candidate improvement**: <what might help, labeled as candidate — not a recommendation>

**Validation**: <how to test whether the improvement helps>
```

### Ranking findings

Rank by consequence first, then by evidence strength:

1. `incorrect_result` > `unverified_completion` > `external_side_effect` > `resource_exhaustion` > `no_impact`
2. `confirmed` > `observed` > `suspected`

A suspected high-consequence finding ranks above a confirmed low-consequence
one — but label it clearly as suspected.

### When the answer is "no supported finding"

```markdown
### Finding: No supported finding

**Assessment**:
The trace does not contain evidence supporting a specific finding for the
question "<question>". <What was checked and why it was insufficient.>

**Suggested next action**: <what additional evidence would help, or why
further analysis is unlikely to be productive>
```

This is not a failure. Not every trace has a problem, and not every question
has a traceable answer. Manufacturing findings to justify the analysis is
worse than reporting nothing.

## Principles

1. **Separate observed fact, possible explanation, and suggested
   intervention.** The Evidence section states facts. The Assessment section
   interprets them. The Candidate improvement section suggests what might
   help. Do not blur these layers.

2. **Every finding needs evidence.** Reference specific run IDs, metric
   values, or quoted text from the trace. A finding without evidence is a
   speculation, not a finding.

3. **Name the pattern before inferring cause.** Use canonical labels from
   `references/mechanisms.md`. Write `tool.ignored_feedback: the agent
   continued after a 403`, not "the model was careless." Then record
   `fault_locus` only if review supports it.

4. **Distinguish trajectory from run quality.** An agent taking a complex
   approach is not the same as a step failing. Don't conflate them.

5. **"No visible verification" is different from "the result is incorrect."**
   State which one the evidence supports. An agent may produce a correct
   result without explicit verification, or an incorrect result despite
   verification. Don't conflate absence of verification with absence of
   correctness.

6. **Weigh against the triangle.** Every finding implies a trade-off
   between quality, cost, and speed. State which axis is affected and whether
   the trade-off seems worth it — but defer the final call to a human
   reviewer.

7. **Check the sanitization report first.** If `complete: false`, the
   anonymization may be incomplete — be careful about quoting trace content.

## Weighing findings: the quality/cost/speed triangle

Every finding should be weighed against the three axes of the improvement
triangle: **quality**, **cost**, **speed**. These are in tension — context
optimization can improve cost and speed while hurting quality, or while
improving it.

See [AGENTS.md — The improvement triangle](../../AGENTS.md) for the full
framing. The key takeaway: take a step back before judging whether a local
optimization helped or harmed the whole task. The CLI surfaces signals; the
analyst decides whether the trade-off was worth it.

## Mechanism vocabulary

See `references/mechanisms.md` for the full vocabulary of trace-observable
patterns, fault loci, impact values, evidence status levels, and
discouraged terms. Use these canonical labels in the **Pattern** field of
each finding.

## Diagnostic benchmark

A set of 5 synthetic benchmark traces with known issues is available in
`tests/fixtures/benchmark/`. These traces are used to evaluate whether
analysis skills help or hurt diagnosis:

- `ignored_tool_error` — agent ignores a 403 and claims success
- `unverified_completion` — agent claims success without verifying
- `legitimate_repetition` — agent reads a file twice because it changed (should NOT be flagged)
- `lost_constraint_after_compaction` — a constraint disappears and is violated
- `clean_execution` — a successful trace (false-positive control)

The benchmark measures: correct issue detection, correct evidence citation,
avoidance of false accusations on clean traces, and distinction between
observation and speculation. See `tests/fixtures/benchmark/spec.py` for the
full spec.
