---
name: navigate-traces
description: Top-down workflow for analyzing AI-agent traces with self-improve CLI. Use when inspecting LangSmith traces, navigating L0-L3 representations, or following the recommended trace analysis workflow.
---

# Navigation Workflow

The recommended top-down workflow for analyzing a trace.
Start cheap, drill down only when needed.

## Quick start

```bash
self-improve init              # Create .env from .env.example (first time)
self-improve list --limit 10   # L0: discover recent traces
self-improve fetch <trace_id>  # Download, anonymize, and build all representations
```

## Granularity levels

### L0 — Discovery

```bash
self-improve list                    # Recent root runs (one per trace)
self-improve list-runs --type llm    # Recent LLM runs across all traces
self-improve list-runs --type tool   # Recent tool runs
```

Use L0 to find interesting traces: errors, high token counts, long latencies.

### L1 — Structure and metrics

```bash
self-improve skeleton <trace_id>        # One line per significant run
self-improve tool-metrics <trace_id>    # Tool call patterns and efficiency
self-improve context-metrics <trace_id>  # Token decomposition and growth
```

L1 gives you the shape of the trace without the content. Use it to decide
where to focus.

### L2 — Narrative

```bash
self-improve narrative <trace_id>           # Compact (default): per-index diff
self-improve narrative <trace_id> --full    # Full: common-prefix diff
```

The narrative is the main analysis input. It tells the chronological story
of the trace with message deltas (only new messages are shown per LLM step)
and tool call arguments/results.

Two modes are available:

- **Compact** (default) — compares each message at its index independently.
  Identical messages collapse to `(N unchanged messages)` even when earlier
  messages changed (e.g. a dynamic system prompt that changes every step).
  This is the best mode for agent consumption — it minimizes tokens.
- **Full** — uses a common-prefix walk that stops at the first mismatch.
  When the system message changes, all subsequent messages are re-dumped.
  Better for humans skimming start-to-end.

When you run `self-improve fetch`, both `narrative_compact.md` and
`narrative_full.md` are saved to `data/ter/<trace_id>/`.

### L3 — Run detail

```bash
self-improve run-detail <trace_id> <run_id>
```

Full untruncated context of a single run. Use this when L2 is insufficient
and you need to see the exact prompts, outputs, or tool arguments of one run.

### Info

```bash
self-improve info <trace_id>
```

Shows sanitization status, run count, and the sanitization report. Always
check this after fetching to verify anonymization ran correctly.

### Prompts

```bash
self-improve prompt pull <name> [--tag TAG]  # Download a prompt from LangSmith
self-improve prompt list                     # List locally saved prompts
self-improve prompt show <name> [--tag TAG]  # Show a saved prompt
self-improve prompt diff <name> <trace_id>   # Compare prompt vs what trace used
```

Prompts are saved as local `.md` files under `data/prompts/`. The agent can
edit them freely — they are local copies, not connected to LangSmith.

### Target agent context

```bash
self-improve ati list          # List registered target agents
self-improve ati show <name>   # Show an agent's architecture document
```

ATI documents are created by the `document-ati` skill and stored under
`data/ati/<name>/architecture.md`.

## Workflow principles

1. **Start at L0, drill down.** Don't jump to L3 without checking L1 first.
2. **Read the skeleton before the narrative.** The skeleton tells you which
   runs matter; the narrative tells you what happened in them.
3. **Use metrics to find signals, not conclusions.** Tool metrics and context
   metrics are deterministic facts. Interpreting them is the analyst's job.
4. **Check the sanitization report.** If `complete: false` or
   `recognizer_version: regex-fallback`, anonymization may be weaker than
   expected. Review outputs carefully.
5. **Stay token-efficient.** The narrative is already compressed (~100-250x
   smaller than raw). Don't dump raw traces into an agent's context when the
   narrative or skeleton will do.

## Privacy

- Traces are anonymized on fetch by default.
- Raw retention (`--keep-raw`) is explicit and local-only.
- The sanitization report shows what was detected and replaced.
- Anonymization is defense in depth, not a guarantee. Review before sharing.
