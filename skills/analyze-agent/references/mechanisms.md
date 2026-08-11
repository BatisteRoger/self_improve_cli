# Mechanism vocabulary for trace observations

A controlled vocabulary for labeling what went wrong in a trace.
Use these labels in the `primary_pattern` and `secondary_patterns` fields
of each observation.

This vocabulary is designed to evolve. As AI agents evolve and new failure
patterns emerge, add new labels, refine definitions, or deprecate terms.
This is a living reference — edit it when you learn something new.

## How to use this vocabulary

Each observation records multiple axes, not a single label:

| Field | Question | Examples |
| --- | --- | --- |
| `primary_pattern` | What visibly went wrong? | `tool.ignored_feedback` |
| `secondary_patterns` | Other patterns present? | `control.nonprogress_loop` |
| `fault_locus` | Where to investigate repair? | `model`, `agent_harness`, `context` |
| `impact` | What was the consequence? | `incorrect_result`, `resource_exhaustion` |
| `evidence_status` | How certain is the diagnosis? | `observed`, `suspected`, `confirmed` |

**Name the trace-observable pattern before inferring cause.**
Write `tool.ignored_feedback: the agent continued after a 403 response`,
not "the model was careless." Then record `fault_locus=model` only if
review supports it.

## Fault loci

Where to investigate the repair. Always provisional unless confirmed.

| Value | Meaning |
| --- | --- |
| `model` | The LLM's reasoning or output |
| `agent_harness` | Prompts, routing, retry logic, context construction |
| `context` | What the model was given (context window, messages) |
| `memory` | Persistent memory (if applicable) |
| `tool_integration` | Tool wrappers, adapters, schemas |
| `external_environment` | Upstream services, APIs, filesystem |
| `task_spec` | The task definition itself |

## Impact values

| Value | Meaning |
| --- | --- |
| `incorrect_result` | Wrong output or answer |
| `unverified_completion` | Agent claims success without verification |
| `external_side_effect` | Consequential action taken |
| `resource_exhaustion` | Tokens, time, or calls consumed excessively |
| `no_impact` | Observed but no consequence (informational) |

## Evidence status

| Value | Meaning |
| --- | --- |
| `observed` | Seen in the trace, not yet reproduced |
| `suspected` | Pattern matches but alternative explanations exist |
| `confirmed` | Verified against raw trace or reproduced |

## Trace-observable labels

### Intent and control

| Label | Definition | Do not conflate with |
| --- | --- | --- |
| `intent.constraint_violation` | Explicit instruction or constraint was not met | `intent.partial_completion` (omitted work, not violated constraint) |
| `intent.partial_completion` | Required subtask remains undone | `control.premature_completion` (agent declared success) |
| `control.premature_completion` | Agent declares success before sufficient completion or verification | Satisficing — use only in research notes |
| `control.unconfirmed_action` | Agent makes a consequential decision it should have confirmed first | Over-initiative, excessive agency |
| `control.nonprogress_loop` | Agent repeats an action/subtask without making progress. Record `loop_trigger` and `iterations` | `recovery.failed_recovery` (recovery attempt, not a loop) |
| `context.goal_drift` | Behavior shifts away from the governing objective over the trace | `intent.scope_change` (owner changed the task) |

### Reasoning and observation

| Label | Definition | Do not conflate with |
| --- | --- | --- |
| `reasoning.inference_error` | Wrong conclusion drawn from available evidence | Hallucination — use `output.ungrounded_claim` instead |
| `reasoning.plan_error` | Unsound or ineffective execution plan | `reasoning.inference_error` (wrong step, not wrong plan) |
| `observation.missed_evidence` | Relevant information was available but not noticed or used | Hallucination — the info was there, just ignored |
| `output.ungrounded_claim` | Claim unsupported by available evidence | Hallucination — be specific about what kind |
| `output.fabricated_completion` | Completion or success claimed without supporting work | `control.premature_completion` (no work vs premature stop) |

### Tools and environment

| Label | Definition | Do not conflate with |
| --- | --- | --- |
| `tool.invalid_arguments` | Arguments violate the tool's schema or format | `tool.incorrect_selection` (right tool, wrong args vs wrong tool) |
| `tool.incorrect_selection` | Wrong tool selected for the task | `tool.nonexistent_reference` (wrong vs absent) |
| `tool.nonexistent_reference` | Agent calls a tool absent from the provided schema | Hallucination — be specific |
| `tool.ignored_feedback` | Tool returned a salient signal/error, agent continued as if it hadn't | `recovery.failed_recovery` (ignored vs failed to recover) |
| `recovery.failed_recovery` | Recoverable anomaly not handled. Record `recovery_mode`: `gave_up`, `blind_retry`, `trusted_corrupt_output` | `environment.external_service_failure` (failure vs non-recovery) |
| `integration.translation_error` | Wrapper/adapter corrupts action or observation across the model-tool boundary | `tool.invalid_arguments` (adapter vs model error) |
| `environment.external_service_failure` | Upstream service failed (error, timeout, rate limit) | Agent failure — don't blame the model |

### Context

| Label | Definition | Do not conflate with |
| --- | --- | --- |
| `context.compaction_loss` | Context compaction retained actions but dropped the rationale/constraint that made them correct. Record `lost_element`: `rationale`, `constraint`, `approval`, `state` | `context.goal_drift` (loss vs drift) |

## Discouraged terms

These are too ambiguous for root-cause labels. Use the specific label instead.

| Discouraged | Use instead |
| --- | --- |
| `hallucination` | `output.ungrounded_claim`, `tool.nonexistent_reference`, `output.fabricated_completion` |
| `leakage` | Be specific: `metadata_exploitation`, `privacy_exposure`, `context_exposure` |
| `drift` | `context.goal_drift` (only with evidence of original goal) |
| `failure` | Name the specific pattern: `tool.ignored_feedback`, `recovery.failed_recovery`, etc. |
| `overfitting` | `update.holdout_overfitting` (for updates) or `evaluation.visible_check_overfitting` (for eval) |
| `satisficing` | `control.premature_completion` |
| `excessive agency` | `control.unconfirmed_action` or `control.unconfirmed_irreversible_action` |

## Mapping from deterministic signals

When the CLI's deterministic metrics flag a signal, use the corresponding
canonical label as the `primary_pattern`:

| Deterministic signal | Canonical label | Notes |
| --- | --- | --- |
| Oscillation (A→B→A alternation) | `control.nonprogress_loop` | Add `loop_trigger` and `iterations` |
| Failed-command retry (same call after error) | `recovery.failed_recovery` | Add `recovery_mode=blind_retry` |
| Repeated calls on same target (>3) | `tool.low_signal_arguments` or `control.nonprogress_loop` | Depends on whether the calls advance |
| Context jump (>5K tokens between steps) | `context.compaction_loss` or informational | Only if a compaction event occurred |
| Dead context ratio > 40% | Informational — no canonical label | Points to `context.*` investigation |
| Monotonic context growth | Informational — no canonical label | Points to `context.*` investigation |

## Adding new labels

When you observe a pattern not covered by this vocabulary:

1. Check if it fits under an existing category (`tool.*`, `control.*`, etc.)
2. Define the label with: canonical id, definition, inclusion criteria, what it is NOT
3. Add it to this file
4. Use it in observations and note whether it recurs

This vocabulary grows with experience. A pattern seen once is a hypothesis;
the same `primary_pattern` seen across traces is a confirmed mechanism.
