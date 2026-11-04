# System Architecture: Agentic RAG Engine

The **Agentic RAG Engine** is an evidence-driven, autonomous multi-step knowledge research architecture designed to solve multi-document queries, cross-document comparative analysis, multi-hop reasoning, and reliable abstention under missing information.

---

## 1. High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                  USER QUERY                                       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        PHASE 4: QUERY ANALYZER & CLASSIFIER                       |
|  - Classify Intent: Simple, Multi-Doc, Comparative, Multi-Hop, Ambiguous, Out-of-Scope  |
|  - Decompose into Targeted Subqueries with Dependency Graphs                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        PHASE 6: RETRIEVAL PLANNER                                 |
|  - Select Optimal Tools (Dense, Sparse, Hybrid, Metadata, Multi-Query)            |
|  - Generate Execution Plan with Per-Subquery Tool Assignments                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        PHASE 5: CONTROLLED TOOL REGISTRY                          |
|  +--------------------+  +--------------------+  +--------------------+          |
|  |  Tool A: Dense     |  |  Tool B: Sparse    |  |  Tool C: Hybrid    |          |
|  |  Vector Index      |  |  BM25 Matching     |  |  RRF Fusion        |          |
|  +--------------------+  +--------------------+  +--------------------+          |
|  +--------------------+  +--------------------+  +--------------------+          |
|  |  Tool D: Metadata  |  |  Tool E: Multi-Q   |  |  Tool F: Reranker  |          |
|  |  Facet Filters     |  |  Subquery Fan-out  |  |  Cross-Encoder/Lex |          |
|  +--------------------+  +--------------------+  +--------------------+          |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        PHASE 7: EVIDENCE EVALUATION & GRADING                     |
|  - Deterministic Overlap & Relevance Scoring                                      |
|  - Coverage Matrix: Subquery Saturation & Missing Entity Detection               |
|  - Decision Matrix: Relevant | Partial | Insufficient | Contradictory | None      |
+-------------------+-------------------------------------+-------------------------+
                    |                                     |
    [Evidence Insufficient & Budget > 0]                  | [Evidence Sufficient OR
                    |                                     |  Budget Exhausted]
                    v                                     v
+---------------------------------------+   +---------------------------------------+
|    PHASE 8: BOUNDED RE-RETRIEVAL      |   |      PHASE 9: ANSWER SYNTHESIS        |
| - Query Reformulation / Expansion     |   | - Strictly Context-Grounded Drafting  |
| - Method Switching (Sparse <-> Dense) |   | - Direct Claim-to-Evidence Attachment |
| - Max Revisions: 2, Max Tool Calls: 6 |   | - Explicit Abstention if Unsupported  |
+-------------------+-------------------+   +-------------------+-------------------+
                    |                                           |
                    +-------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    PHASE 10: CITATION & VERIFICATION ENGINE                       |
|  - Validate Cited IDs Against Retrieved Evidence Index                            |
|  - Hallucination / Unsupported Assertion Detection                                |
|  - Abstention Enforcement: If Grounding < Threshold -> Abstain or Strip Claim     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         FINAL VERIFIED AGENT RESPONSE                             |
|  - Answer, Structured Citations, Confidence Score, Full Execution Trace           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Principles

1. **Evidence-Driven State Machine**:
   Every state transition is discrete, inspectable, and serialized via Pydantic models. Uncontrolled recursive prompting or unbounded looping is structurally prevented.

2. **Defense-in-Depth for Untrusted Documents**:
   Retrieved text is treated strictly as **untrusted data**. Strict delimiters, prompt injection guards, and semantic isolation layers prevent data poisoning from taking over agent control flow.

3. **Multi-Strategy Hybrid Retrieval**:
   Combining dense vector search (capturing semantic analogies and conceptual matches) with sparse lexical BM25 (capturing exact keywords, domain identifiers, and numbers) via Reciprocal Rank Fusion (RRF).

4. **Calibrated Verification Before Output**:
   Answers are not considered complete until every cited sentence is verified against actual chunk text in the retrieval cache. Unsupported claims trigger automated trimming or explicit abstention.

