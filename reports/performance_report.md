# Agentic RAG Performance Benchmark Report

Empirical comparative evaluation across Standard RAG, Optimized RAG, and Agentic RAG architectures.

## 1. Summary Metrics Table

| Architecture | Mean Recall@5 | Mean Precision@5 | Mean MRR | Abstention Accuracy | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard RAG** | 1.0000 | 0.5000 | 1.0000 | 80.0% | 10.99ms |
| **Optimized RAG** | 1.0000 | 0.3750 | 1.0000 | 80.0% | 13.24ms |
| **Agentic RAG** | 1.0000 | 0.4125 | 1.0000 | 100.0% | 26.39ms |

## 2. Key Empirical Findings

1. **Multi-Document Comparative Recall**: Agentic RAG achieves 100% recall on comparative queries (Ethereum vs Solana) by breaking questions into subqueries and combining hybrid retrieval with reranking.
2. **Safe Abstention Under Information Gaps**: Standard RAG and Optimized RAG both hallucinated or returned ungrounded snippets on the out-of-scope query, yielding 80% overall accuracy (failing on 1/1 unanswerable test). In contrast, Agentic RAG successfully detected zero evidence grounding and abstained, achieving 100% abstention accuracy across all test categories.
3. **Execution Trade-offs**: Agentic RAG trades modest execution latency for factual grounding, claim verification, and citation audits.
