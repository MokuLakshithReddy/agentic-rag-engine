"""
Concrete implementations of registered retrieval tools.
Covers Dense, Sparse, Hybrid, Metadata, Multi-Query, and Neural Cross-Encoder Reranking.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.tools.base import BaseTool
from src.state.state import EvidenceChunk
from src.ingestion.indexer import KnowledgeBase


class DenseRetrievalTool(BaseTool):
    """Tool A: Dense Semantic Vector Search."""

    def __init__(self, kb: KnowledgeBase):
        super().__init__(
            name="dense_retrieval",
            description="Semantic embeddings vector search. Best for conceptual and semantic similarity queries."
        )
        self.kb = kb

    def _execute(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceChunk]:
        return self.kb.dense.search(query=query, top_k=top_k, filter_metadata=filter_metadata)


class SparseRetrievalTool(BaseTool):
    """Tool B: Sparse BM25 Keyword Search."""

    def __init__(self, kb: KnowledgeBase):
        super().__init__(
            name="sparse_retrieval",
            description="BM25 keyword search. Best for exact token matches, acronyms, IDs, and technical jargon."
        )
        self.kb = kb

    def _execute(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceChunk]:
        return self.kb.sparse.search(query=query, top_k=top_k, filter_metadata=filter_metadata)


class HybridRetrievalTool(BaseTool):
    """Tool C: Hybrid Dense + Sparse Search via Reciprocal Rank Fusion."""

    def __init__(self, kb: KnowledgeBase):
        super().__init__(
            name="hybrid_retrieval",
            description="Fused semantic and keyword retrieval using Reciprocal Rank Fusion. Best overall default."
        )
        self.kb = kb

    def _execute(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceChunk]:
        return self.kb.hybrid.search(query=query, top_k=top_k, filter_metadata=filter_metadata)


class MetadataSearchTool(BaseTool):
    """Tool D: Structured Faceted Metadata Filtering."""

    def __init__(self, kb: KnowledgeBase):
        super().__init__(
            name="metadata_search",
            description="Faceted search filtering by doc_id, category, source file, or structured metadata attributes."
        )
        self.kb = kb

    def _execute(
        self,
        query: Optional[str] = None,
        doc_id: Optional[str] = None,
        category: Optional[str] = None,
        source_file: Optional[str] = None,
        custom_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[EvidenceChunk]:
        return self.kb.metadata.search(
            doc_id=doc_id,
            category=category,
            source_file=source_file,
            custom_filters=custom_filters,
            query=query,
            top_k=top_k
        )


class MultiQueryRetrievalTool(BaseTool):
    """Tool E: Multi-Query Subquery Fan-out & Fusion."""

    def __init__(self, kb: KnowledgeBase):
        super().__init__(
            name="multi_query_retrieval",
            description="Executes a collection of targeted subqueries across the hybrid index and merges unique results."
        )
        self.kb = kb

    def _execute(
        self,
        subqueries: List[str],
        top_k_per_query: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceChunk]:
        all_hits: Dict[str, EvidenceChunk] = {}
        for sq in subqueries:
            hits = self.kb.hybrid.search(query=sq, top_k=top_k_per_query, filter_metadata=filter_metadata)
            for h in hits:
                if h.chunk_id not in all_hits or h.score > all_hits[h.chunk_id].score:
                    h.strategy = "multi_query_retrieval"
                    all_hits[h.chunk_id] = h

        sorted_hits = sorted(all_hits.values(), key=lambda c: c.score, reverse=True)
        return sorted_hits


class RerankEvidenceTool(BaseTool):
    """Tool F: Neural Cross-Encoder Candidate Evidence Reranking."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(
            name="rerank_evidence",
            description="Re-scores candidate evidence chunks using neural cross-attention between query and passage."
        )
        self.model_name = model_name
        self._cross_encoder = None
        self._load_cross_encoder()

    def _load_cross_encoder(self) -> None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            self._cross_encoder = CrossEncoder(self.model_name)
        except Exception:
            self._cross_encoder = None

    def _execute(
        self,
        query: str,
        candidates: List[EvidenceChunk],
        top_k: int = 5
    ) -> List[EvidenceChunk]:
        if not candidates:
            return []

        # Neural Cross-Encoder scoring if available
        if self._cross_encoder is not None:
            try:
                pairs = [[query, c.content] for c in candidates]
                raw_scores = self._cross_encoder.predict(pairs)
                # Normalize cross-encoder scores via sigmoid
                norm_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))

                scored: List[EvidenceChunk] = []
                for idx, c in enumerate(candidates):
                    scored.append(
                        EvidenceChunk(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id,
                            content=c.content,
                            score=round(float(norm_scores[idx]), 4),
                            strategy="neural_cross_encoder",
                            page_number=c.page_number,
                            metadata=c.metadata
                        )
                    )
                scored.sort(key=lambda x: x.score, reverse=True)
                return scored[:top_k]
            except Exception:
                pass

        # High-precision semantic & lexical cross-alignment fallback
        q_tokens = set(query.lower().split())
        scored_fallback: List[EvidenceChunk] = []

        for c in candidates:
            c_tokens = set(c.content.lower().split())
            overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))
            rerank_score = round(0.65 * c.score + 0.35 * overlap, 4)

            scored_fallback.append(
                EvidenceChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    content=c.content,
                    score=rerank_score,
                    strategy="rerank_evidence",
                    page_number=c.page_number,
                    metadata=c.metadata
                )
            )

        scored_fallback.sort(key=lambda x: x.score, reverse=True)
        return scored_fallback[:top_k]
