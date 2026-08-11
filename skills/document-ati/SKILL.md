---
name: document-ati
description: Create or update a target agent's architecture document (ATI) for trace analysis context. Use when setting up a new target agent for analysis, or when the agent's architecture has changed and the context document needs updating.
---

# Document a target agent (ATI)

Create an architecture document for the Agent-To-Improve (ATI) so that trace
analysis happens in context. Without this, traces are analyzed in a vacuum.

This document is most valuable when you have access to the agent's source code
or detailed specs. The architecture context is what turns "the agent called
search 7 times" into "the agent's retrieval strategy is inefficient because
it doesn't consolidate results." Without code or specs access, you can still
create a useful document from the agent's documentation, README, or by
interviewing the developer — but it will be less precise.

If you have code access, read the agent's entry point, tool definitions, and
prompt templates before writing the document. If you don't, ask the user for
as much context as they can provide.

## What to include

Write the document as `data/ati/<agent_name>/architecture.md` with these sections:

### Purpose

What the agent is supposed to do. One or two sentences.

Example: "Answer customer questions about orders by retrieving order data
from the database and composing a response."

### Tools

List every tool the agent can call, with:
- Tool name
- What it does (one line)
- What it returns (data shape, not full schema)
- Known limitations or edge cases

### Prompt structure

Describe the system prompt's structure at a high level:
- Role definition
- Instructions (ordered or grouped)
- Guardrails or constraints
- Output format requirements

Do not paste the full prompt — describe its structure. The actual prompt
can be pulled with `self-improve prompt pull <name>` if needed.

### Known limitations

What the agent struggles with, based on prior analysis or known issues:
- Specific inputs that cause failures
- Tools that are unreliable
- Edge cases in the prompt

### Expected behavior

What a good trace looks like for this agent:
- Typical number of steps
- Expected tool call sequence
- Expected token usage range
- Expected latency range

This is the baseline against which traces are compared.

## How to create it

1. **Gather context.** If you have code access, read the agent's entry point,
   tool definitions, and prompt templates. Otherwise, ask the user for docs,
   specs, or a description of how the agent works.
2. Draft the document using the sections above.
3. Save to `data/ati/<agent_name>/architecture.md`.
4. Reference it when analyzing traces:

```bash
self-improve ati show <agent_name>
```

## Privacy

- ATI documents may contain internal architecture details.
- They are stored under `data/` (gitignored) — do not commit them.
- Do not include credentials, API keys, or customer data in the document.
- If the agent's prompt contains sensitive logic, describe it at a high level
  rather than copying it verbatim.

## When to update

- After a prompt change (pull the new prompt and update the structure section)
- After a tool is added or removed
- After a known limitation is resolved or discovered
- Before analyzing traces from a new version of the agent
