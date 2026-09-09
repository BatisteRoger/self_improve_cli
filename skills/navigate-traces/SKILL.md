---
name: navigate-traces
description: Choose the cheapest self-improve CLI view that answers a specific trace-analysis question, and stop when the question is answered. Use when deciding which CLI command answers a question about a LangSmith trace already fetched locally — not for fetching, listing, or general exploration.
---

# Navigate traces by question

The `self_improve` CLI exposes bounded, recomputable views over a fetched
trace. This skill helps you pick the cheapest view that answers your
question, then stop. It does not replace your own reading of the trace —
if a view already gives you the answer, do not run more commands to fill
a template.

## How to use this skill

1. **Name the question precisely.** "Why did it fail?" is too broad.
   "Why did the `edit_file` call at step 7 fail?" is precise enough to
   pick a view.
2. **Pick the cheapest view that can answer it** from the table below.
3. **Run exactly that view.** Read its output before deciding whether to
   drill down.
4. **Stop** as soon as the question is answered with cited run IDs, or as
   soon as you can state that the evidence is unavailable.

## Question → view

| Question | Start with | Next step only if insufficient |
|----------|-----------|--------------------------------|
| "What happened in this trace?" | `skeleton` (shows assessment header if set) | `narrative` for the relevant section |
| "What was the agent asked to do, and did it succeed?" | `assess <trace_id>` (show) | `skeleton` for run-level evidence |
| "Which steps errored or were cancelled?" | `skeleton --errors-only` | `error-neighborhood` for context around each error |
| "Why did this tool call fail?" | `run-detail` on the tool run | `context-at` for the next model input |
| "Did the model receive that tool result/error?" | `context-at` for the step | `run-detail --inputs-only` on the next LLM run |
| "Why did it read this file 4 times?" | `tool-metrics` | `target-timeline <trace_id> <file> --compact` (overview), then `run-detail` on each touch |
| "Did compaction drop the user's constraint?" | `context-metrics` (look for a drop) | `run-detail` on the LLM steps before/after the drop |
| "What supports the final claim?" | `narrative` (find the claim) | `run-detail --outputs-only` on the last LLM run |
| "Where did this become expensive?" | `context-metrics` growth curve | `tool-metrics` for attribution, then `run-detail` on the largest contributor |
| "Is this trace clean?" | `skeleton --errors-only` (quick triage) | `tool-metrics` and `context-metrics` for redundancy/anomalies, `error-neighborhood` for error recovery |
| "Did the agent recover from that error?" | `error-neighborhood` | `run-detail` on the error and the next step |
| "What should I inspect next?" | The signal you already have (a metric, a run ID) | A specific run reference, not a generic checklist |

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

When you have your evidence, write a findings report using the format
described in the `analyze-agent` skill (Finding, Evidence, Assessment,
Likely locus, Suggested next action, Candidate improvement, Validation).

## Reference: the view ladder

When you have no precise question and need a broad scan of an unfamiliar
trace, the L0 → L1 → L2 → L3 ladder is a sensible default. It is not a
requirement — skip levels when you already know where to look.

- **L0 — Discovery.** `list`, `list-runs --type llm|tool`. Find traces by
  error, token count, or latency.
- **L1 — Structure and metrics.** `skeleton`, `tool-metrics`,
  `context-metrics`, `skill-metrics`. The shape of the trace without the
  content. Use it to decide where to focus.
- **L2 — Narrative.** `narrative` (compact by default; `--full` for
  humans skimming). The chronological story with message deltas and tool
  results.
- **L3 — Run detail.** `run-detail <trace_id> <run_id>`. Full
  untruncated context of a single run. Use when L2 is insufficient.
  `context-at <trace_id> <step>` is a bounded, selective alternative
  that shows what the model saw at a given main-loop step, with
  `--from N --to M` for diffs, `--inputs-only`, and `--tool <id>` to
  isolate one tool result.

`compare <trace_a> <trace_b>` sits beside the ladder for A/B testing two
traces (tokens, latency, tool calls, skills).

## Principles

- **Start cheap, drill down only when needed.** The narrative is already
  compressed (~100–250x smaller than raw). Don't dump raw traces into an
  agent's context when the narrative or skeleton will do.
- **Metrics are facts, not conclusions.** Tool, context, and skill
  metrics are deterministic. Interpreting them is the analyst's job.
- **Cite run IDs.** Every finding should be traceable to a specific run.
- **Check the sanitization report.** If `complete: false` or
  `recognizer_version: regex-fallback`, anonymization may be weaker than
  expected. Review outputs carefully before sharing.

## Privacy

- Traces are anonymized on fetch by default.
- Raw retention (`--keep-raw`) is explicit and local-only.
- The sanitization report shows what was detected and replaced.
- Anonymization is defense in depth, not a guarantee. Review before sharing.
