---
name: analyze-agent
description: Systematic trace analysis workflow — read skeleton, check metrics for signals (repeated tools, context jumps, dead context), read narrative, drill into runs, produce structured observations. Use when analyzing an AI agent's execution trace to identify improvement opportunities.
---

# Analyze an agent trace

Systematic workflow for interpreting a trace after it has been fetched.
Produces structured observations, not conclusions — interpretation stays with the analyst.

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

## Step 1: Read the skeleton (L1)

```bash
self-improve skeleton <trace_id>
```

Look for:
- Runs with errors (status != success)
- Runs with high token counts (prompt or completion)
- Long latency gaps between runs
- Unexpected run types or names (middleware, retries, duplicate calls)

The skeleton tells you which runs matter. Note the run IDs worth investigating.

## Step 2: Check tool metrics (L1)

```bash
self-improve tool-metrics <trace_id>
```

Signals to look for:
- **Repeated calls on the same target** (>3 calls to the same tool with similar args — possible redundant work)
- **Low modify-call granularity** (tools that make tiny changes — possible inefficiency)
- **High token attribution** (tools that dominate context growth — candidates for output trimming)

Known limitation: tools that wrap nested LLM calls show 0 attributed tokens.
Cross-reference with the skeleton — look for tool runs with child LLM runs.

## Step 3: Check context metrics (L1)

```bash
self-improve context-metrics <trace_id>
```

Signals to look for:
- **Context jumps** (>5K token growth between steps — possible unbounded tool output)
- **Dead context ratio > 40%** (most of the context is not contributing to the output — possible prompt bloat or irrelevant history)
- **Monotonic growth** (context only grows, never shrinks — no summarization or pruning)

These are deterministic facts. They do not by themselves prove a problem —
they point to where to look.

## Step 4: Read the narrative (L2)

```bash
self-improve narrative <trace_id>
```

The narrative tells the chronological story with message deltas (only new
messages per LLM step) and tool call arguments/results. This is the main
analysis input.

Look for:
- Loops or cycles (the agent repeating the same reasoning or tool calls)
- Tool calls that return errors the agent doesn't handle
- Messages that don't advance toward the goal
- Prompt instructions that are ignored or misinterpreted

## Step 5: Drill into specific runs (L3)

```bash
self-improve run-detail <trace_id> <run_id>
```

Use L3 only when L2 is insufficient and you need the exact prompts, outputs,
or tool arguments of one specific run. This is expensive — don't dump it
into context unless you've identified the run as worth investigating.

## Step 6: Produce structured observations

Write observations as Markdown using a multi-axis schema.
Each observation labels **what happened** before inferring **why**.

See `references/mechanisms.md` for the full vocabulary, fault loci, impact
values, evidence status levels, and discouraged terms.

### RunEO — single run observation

```markdown
## RunEO: <run_id>

**Run**: <name> (<run_type>)
**primary_pattern**: <canonical label from mechanisms.md>
**secondary_patterns**: <optional, other patterns present>
**fault_locus**: <model | agent_harness | context | memory | tool_integration | external_environment | task_spec>
**impact**: <incorrect_result | unverified_completion | external_side_effect | resource_exhaustion | no_impact>
**evidence_status**: <observed | suspected | confirmed>

**Observation**: <what happened, stated factually — name the trace-observable pattern>

**Evidence**: <quote from the trace — message, tool call, or metric value, with run id>

**Hypothesis**: <why it might have happened, labeled as hypothesis>

**Suggested investigation**: <what to check next, not a fix>
```

### TraceEO — whole trace observation

```markdown
## TraceEO: <trace_id>

**Trace**: <trace_id> (<N> runs, <duration>)
**primary_patterns**: <canonical labels that characterize the trace>
**fault_loci**: <where to investigate, provisional>
**evidence_status**: <observed | suspected | confirmed>

**Observation**: <what the trace reveals about the agent's behavior>

**Evidence**: <references to specific runs or metric values>

**Hypotheses**: <why patterns might have occurred, labeled as hypotheses>

**Suggested investigations**: <what to check next, not fixes>
```

## Weighing observations: the quality/cost/speed triangle

Every observation should be weighed against the three axes of the
improvement triangle: **quality**, **cost**, **speed**. These are in
tension — context optimization can improve cost and speed while hurting
quality, or while improving it.

See [AGENTS.md — The improvement triangle](../../AGENTS.md) for the full
framing. The key takeaway: take a step back before judging whether a
local optimization helped or harmed the whole task. The CLI surfaces
signals; the analyst decides whether the trade-off was worth it.

## Principles

1. **Observations, not conclusions.** State what happened. Hypotheses are
   labeled as hypotheses. Fixes are out of scope — a separate evaluator or
   human reviewer decides what to change.
2. **Every observation needs evidence.** Reference specific run IDs, metric
   values, or quoted text from the trace.
3. **Name the pattern before inferring cause.** Use canonical labels from
   `references/mechanisms.md`. Write `tool.ignored_feedback: the agent
   continued after a 403`, not "the model was careless." Then record
   `fault_locus` only if review supports it.
4. **Distinguish trajectory from run quality.** An agent taking a complex
   approach is not the same as a step failing. Don't conflate them.
5. **Weigh against the triangle.** Every observation implies a trade-off
   between quality, cost, and speed. State which axis is affected and
   whether the trade-off seems worth it — but defer the final call to a
   human reviewer.
6. **Check the sanitization report first.** If `complete: false`, the
   anonymization may be incomplete — be careful about quoting trace content.
