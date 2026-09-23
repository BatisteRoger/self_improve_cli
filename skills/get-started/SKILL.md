---
name: get-started
description: First-read onboarding for an analyst agent — self-assess which capabilities your harness provides (shell commands, file writes, running the target agent), verify the local setup with `self-improve doctor`, and pick the right skill for the job. Use when working with this CLI for the first time, or to check whether a full improvement loop is possible in the current environment.
---

# Get started as an analyst agent

The CLI is the interface; your harness provides the capabilities. Before
running analysis, check which tier of work your environment supports —
the honest answer shapes what you can deliver.

## Step 1: Self-check your capabilities

The CLI does not require an agent — a human relaying commands works too —
but each capability below unlocks more of the workflow:

| Capability | Unlocks |
|------------|---------|
| Execute shell commands | Everything — running `self-improve doctor` proves this |
| Write files under `data/` | Persisted assessments and findings — memory across sessions |
| Reach LangSmith (API key configured) | `fetch`/`list`; without it you can only analyze already-fetched traces |
| Trigger a run of the target agent | Step 1 of the improvement loop (`improve-agent`) |
| Apply or hand off changes to the target | Step 5 of the loop — without it you diagnose but cannot close the loop |

State which of these you have before starting. If a capability is missing,
say so — "I can diagnose but not re-run" is a valid scope, and better than
silently skipping loop steps.

## Step 2: Verify the local setup

```bash
self-improve doctor                  # default profile: analysis commands
self-improve doctor --profile fetch  # escalates fetch prerequisites to fail
```

Green (exit 0) means the CLI side is ready. Doctor checks the environment
only — it cannot see your capabilities; only Step 1 covers those.

## Step 3: Pick your skill

| Goal | Skill |
|------|-------|
| Diagnose one trace | `analyze-agent` (evidence via `navigate-traces`) |
| Run an improvement iteration | `improve-agent` — needs the loop capabilities above |
| Set up a new target agent | `document-ati` |
| Decide which view answers a question | `navigate-traces` |

Print any of them with `self-improve skill <name>`.

## If your harness cannot load skills

Everything here works without skill support: print any skill with
`self-improve skill <name>`, run the commands, and use `--help` on any
command. Skills are guidance, not gatekeeping.
