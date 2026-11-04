# Evaluation Framework & Benchmarking Methodology

This document details the rigorous evaluation protocol used to compare **Standard RAG**, **Optimized RAG**, and **Agentic RAG** across retrieval, answer generation, and agentic loop behavior.

---

## 1. Multi-Tier Evaluation Metrics

### A. Retrieval Quality Metrics
* **Recall@K**: Proportion of required reference passages retrieved within top-$K$ candidates.
* **Precision@K**: Proportion of retrieved top-$K$ candidates that contain relevant ground-truth evidence.
* **MRR (Mean Reciprocal Rank)**: Position of the first relevant document retrieved across all evaluation queries.
* **NDCG@K (Normalized Discounted Cumulative Gain)**: Graded ranking relevance metric penalizing relevant documents ranked lower in the result list.

### B. Answer Quality & Grounding
* **Faithfulness / Grounding Score**: Percentage of factual statements in the generated response that can be directly mapped to cited evidence passages.
* **Citation Precision**: Proportion of citations in the output that legitimately support their referencing sentence.
* **Citation Recall**: Proportion of required facts that possess a verifiable citation reference.
* **Abstention Accuracy**: Correct detection and abstention on unanswerable or out-of-corpus queries (penalizing false confidence and hallucinations).

### C. Agentic Behavior & Efficiency
* **Task Completion Rate**: Percentage of queries successfully resolved without unhandled errors.
* **Tool Call Efficiency**: Number of tool calls required per successfully resolved query.
* **Retry Effectiveness**: Success rate of queries that initially lacked sufficient evidence and entered the re-retrieval loop.
* **Latencies & Budgets**: Average latency (milliseconds) and token usage per stage.

---

## 2. Experimental Setup: The Three Paradigms

1. **Standard RAG (Baseline)**:
   - Single-shot dense vector retrieval ($K=3$).
   - Direct prompt injection into LLM generation without query decomposition, planning, or self-reflection.
2. **Optimized RAG (Hybrid + Reranking)**:
   - Dense vector retrieval + BM25 sparse retrieval.
   - Reciprocal Rank Fusion ($K=5$) + Reranker.
   - Static prompt without dynamic re-retrieval or verification loops.
3. **Agentic RAG (Full Autonomous Pipeline)**:
   - Intent classification & subquery decomposition.
   - Adaptive tool selection via `RetrievalPlanner`.
   - Evidence sufficiency evaluation & gap detection.
   - Bounded re-retrieval loop on missing entities.
   - Grounded synthesis & post-generation citation verification with abstention guard.

