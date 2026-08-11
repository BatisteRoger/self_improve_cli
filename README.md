# Self-improve CLI

A small, open-source CLI that turns raw AI-agent traces into compact, progressive representations an analyst agent (or human) can actually use.

## The problem

Observability tools like LangSmith capture detailed execution traces — every LLM call, tool call, retry, and middleware wrapper. A single trace can be 1–2 MB of JSON across dozens of runs. That's too large to feed into an LLM for analysis, and too noisy for a human to scan by hand.

You end up either:

- Staring at the LangSmith UI, clicking through runs one by one.
- Dumping raw JSON into a prompt and paying for thousands of tokens of middleware noise.

## What this CLI does

It compresses raw traces into **token-efficient representations (TER)** at four levels:

| Level | Command | What you get | Typical size |
| ----- | ------- | ------------- | ------------- |
| L0 | `list`, `list-runs` | Discover which traces exist | One line per trace |
| L1 | `skeleton`, `tool-metrics`, `context-metrics` | Structure, metrics, growth signals | ~1–3 KB |
| L2 | `narrative` | Chronological story with message deltas | ~15 KB (~4K tokens) |
| L3 | `run-detail` | Full untruncated content of one run | On demand |

A 1.6 MB trace becomes a ~17 KB narrative — roughly **100–250x smaller** — with the option to drill down to any specific run when you need the full context.

It also **anonymizes by default**: PII and secrets are replaced with stable placeholders before anything is written to disk.

## Why not just use LangSmith directly?

| | LangSmith UI | LangSmith SDK / CLI | Self-improve CLI |
| --- | --- | --- | --- |
| View traces | Click through runs in a web UI | Raw JSON / Run objects | Progressive TER (L0–L3) |
| Token cost for LLM analysis | N/A (human-only) | High (raw JSON, lots of noise) | Low (~250x compression) |
| Privacy | Traces stay in LangSmith | You handle it yourself | Anonymized on ingestion by default |
| Offline analysis | No (requires login) | You build it | Yes — `fetch` once, analyze offline |
| Metrics | Basic token counts | You compute them yourself | Tool patterns, context growth, dead context ratio |
| Provider lock-in | LangSmith only | LangSmith only | Canonical model, LangSmith adapter first |

This CLI is not a replacement for LangSmith — it's a **compression and analysis layer** that sits on top of it. You still use LangSmith to collect traces; this tool helps you and your analyst agent make sense of them.

## Quick start

```bash
pip install self-improve-cli[langsmith]
self-improve init              # Create .env from template
# Edit .env with your LangSmith API key and project name
self-improve skill             # List available skills
self-improve list --limit 10   # L0: discover recent traces
self-improve fetch <trace_id>  # Download, anonymize, and build all representations
self-improve narrative <trace_id>   # L2: read the trace story
self-improve run-detail <trace_id> <run_id>  # L3: drill into one run
```

## Skills

The CLI ships with agent skills in the `skills/` directory. Install them into your agent with:

```bash
npx skills add BatisteRoger/self_improve_cli
```

Available skills:

- **navigate-traces** — the top-down trace analysis workflow (L0 to L3)
- **analyze-agent** — systematic trace interpretation with structured observation templates
- **document-ati** — create architecture documents for target agents being analyzed

You can also print a skill directly from the CLI:

```bash
self-improve skill                 # list available skills
self-improve skill navigate-traces # print a specific skill
```

## Prompts

Pull prompts from LangSmith Prompt Hub and save them as local `.md` files your agent can edit:

```bash
self-improve prompt pull my-agent --tag prod   # Download a prompt
self-improve prompt list                        # List saved prompts
self-improve prompt show my-agent --tag prod    # Show a saved prompt
self-improve prompt diff my-agent <trace_id>    # Compare vs what a trace used
```

Prompts are saved under `data/prompts/` (gitignored). The agent can edit them freely — they're local copies. **No upload capability**: you manually push changes to LangSmith.

## Target agent context

Document the agent you're analyzing so trace interpretation happens in context:

```bash
self-improve ati list          # List registered target agents
self-improve ati show my-agent # Show an agent's architecture document
```

ATI documents are created by the `document-ati` skill and stored under `data/ati/<name>/architecture.md` (gitignored).

## What it does not do

- **Modify the target agent.** This CLI observes and analyzes. It does not change prompts, code, or models.
- **Replace human review.** It produces representations and metrics. A human (or a separate evaluator agent) interprets them and decides what to do.
- **Upload to LangSmith.** The `prompt` command is read-only. Humans have to manually push prompt changes. This is opinionated and we care about this. Please don't make a PR to "solve" this.
- **Guarantee perfect anonymization.** Anonymization is defense in depth. Review output before sharing.

## Where to learn more

- [AGENTS.md](AGENTS.md) — project values, architecture, non-negotiable principles, and working rules.
- [SECURITY.md](SECURITY.md) — threat model, privacy guarantees, and reporting process.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, tests, and review expectations.

## Status

Early release (v0.1.0). LangSmith is the first supported trace source; the architecture is designed for provider interoperability.

## License

[MIT](LICENSE)
