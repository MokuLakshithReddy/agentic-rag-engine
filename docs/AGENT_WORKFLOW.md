# Agent Workflow Specification: Controlled Execution & State Transitions

This document defines the deterministic state machine, lifecycle hooks, and bounded iterative loops implemented by the Agentic RAG Engine.

---

## 1. Execution States and Transition Rules

| State | Responsibility | Input Requirements | Success Next State | Failure / Fallback Next State |
| :--- | :--- | :--- | :--- | :--- |
| **`ANALYZE_QUERY`** | Classify query intent and generate decomposed subqueries. | Raw User Query | `PLAN_RETRIEVAL` | Fallback single-query plan -> `PLAN_RETRIEVAL` |
| **`PLAN_RETRIEVAL`** | Select retrieval tools and assign subqueries. | Query classification & subqueries | `EXECUTE_TOOLS` | Fallback to `hybrid_search` -> `EXECUTE_TOOLS` |
| **`EXECUTE_TOOLS`** | Execute registered tool operations with timeout & budget guards. | Selected tools & subqueries | `EVALUATE_EVIDENCE` | `EVALUATE_EVIDENCE` (with partial evidence) |
| **`EVALUATE_EVIDENCE`** | Grade relevance, coverage, and identify gaps. | Retrieved evidence corpus | If sufficient: `SYNTHESIZE_ANSWER`<br>If insufficient & budget $>0$: `REFORMULATE_QUERY` | If budget exhausted: `SYNTHESIZE_ANSWER` |
| **`REFORMULATE_QUERY`** | Refine missing entity queries, switch retrieval tool, increment revision count. | Identified gaps & remaining budget | `EXECUTE_TOOLS` | `SYNTHESIZE_ANSWER` |
| **`SYNTHESIZE_ANSWER`** | Generate grounded answer or trigger explicit abstention. | Evidence corpus & verification state | `VERIFY_CITATIONS` | `ABSTAIN` |
| **`VERIFY_CITATIONS`** | Audit citation IDs and check factual support. | Draft answer & cited IDs | `FINALIZE_RESPONSE` | Strip ungrounded claims or `ABSTAIN` |
| **`FINALIZE_RESPONSE`**| Assemble verified payload and trace telemetry. | Verified answer and metadata | Terminal State (`COMPLETED`) | Terminal State (`FAILED`) |

---

## 2. Hard Execution Budgets & Loop Bounds

To prevent infinite loops, API runaway costs, and denial-of-service vulnerabilities, the engine enforces non-negotiable execution limits:

* **Max Retrieval Planning Revisions**: `2`
* **Max Cumulative Tool Executions Per Request**: `6`
* **Max Subqueries Per Plan**: `4`
* **Tool Timeout Per Execution**: `5.0 seconds`
* **Max Context Characters Injected to Synthesis**: `12,000 chars`
* **Abstention Threshold**: If evidence relevance score $< 0.35$ or zero relevant chunks found for critical subqueries.

---

## 3. Observability and State Telemetry

Every request generates a unique `request_id` (UUIDv4) and records structured trace steps:

```json
{
  "request_id": "c1f7b9e2-3481-4b11-9a72-6a75f8f8216c",
  "step_index": 3,
  "stage": "EVALUATE_EVIDENCE",
  "timestamp": "2026-11-05T00:01:15.120Z",
  "tool_budget_remaining": 4,
  "decision": "EVIDENCE_INSUFFICIENT",
  "gaps_identified": ["consensus validator threshold details"],
  "next_action": "REFORMULATE_QUERY"
}
```
*Private model internal chain-of-thought is never persisted or logged; only verified inputs, structured metadata, and audited citations are preserved.*

