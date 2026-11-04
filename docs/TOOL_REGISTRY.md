# Tool Registry Specification

All retrieval operations are mediated by a strictly controlled, sandboxed **Tool Registry**. The agent cannot execute arbitrary shell commands, raw filesystem calls, or unmonitored external network operations.

---

## 1. Registered Tool Inventory

| Tool ID | Name | Core Technology | Best Suited For | Input Schema | Output Schema |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dense_retrieval`** | Dense Vector Search | TF-IDF / Neural Embeddings + Cosine Similarity | Conceptual, thematic, and semantic similarity search | `query: str, top_k: int = 5, filter_meta: Optional[dict]` | `List[EvidenceChunk]` |
| **`sparse_retrieval`**| BM25 Lexical Search | BM25Okapi Token Frequency Inverse Document Frequency | Exact acronyms, part numbers, entities, and specific keywords | `query: str, top_k: int = 5` | `List[EvidenceChunk]` |
| **`hybrid_retrieval`**| Hybrid Dense-Sparse Search | Reciprocal Rank Fusion ($k=60$) | Broad multi-part questions needing both semantic and keyword matching | `query: str, top_k: int = 5, dense_weight: float = 0.5` | `List[EvidenceChunk]` |
| **`metadata_search`** | Faceted Metadata Filter | Attribute Filtering + Content Match | Targeted queries filtering by `doc_id`, `category`, `source` | `doc_id: Optional[str], category: Optional[str], top_k: int = 5` | `List[EvidenceChunk]` |
| **`multi_query_retrieval`** | Fan-out Subquery Retriever | Concurrent Subquery Execution + Deduplication | Multi-hop and multi-document comparative questions | `subqueries: List[str], top_k_per_query: int = 3` | `List[EvidenceChunk]` |
| **`rerank_evidence`** | Candidate Reranker | Cross-score Lexical/Semantic Alignment | High-precision reordering of initial evidence candidates | `query: str, candidates: List[EvidenceChunk], top_k: int = 5` | `List[EvidenceChunk]` |

---

## 2. Safety & Sandboxing Guarantees

1. **Explicit Allowlist**: Only tools present in `ToolRegistry.registered_tools` can be invoked.
2. **Timeout Enforcement**: Every tool execution is bounded by an execution timer ($5.0$ seconds). If a tool exceeds the threshold, an empty or cached response is returned without crashing the orchestrator.
3. **External Network Policy**: External internet web search is **disabled by default**. In offline competition mode, only local vector indices and document stores are queried.
4. **Input Sanitization**: Query strings undergo strict input scrubbing (removing null bytes, non-printable characters, and excessive token lengths) before being passed to retrieval indices.

