"""
Hybrid retrieval engine combining Dense semantic vector search and Sparse BM25 keyword matching
using Reciprocal Rank Fusion (RRF) and deduplication.
"""

from typing import List, Dict, Any, Optional
from src.state.state import EvidenceChunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """
    Reciprocal Rank Fusion hybrid retrieval engine.
    Fuses rankings from dense vector and sparse lexical retrievers.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceChunk]:
        """
        Executes hybrid dense-sparse search and merges rankings via RRF.
        """
        k = top_k or self.top_k
        pool_k = k * 3

        dense_hits = self.dense_retriever.search(query, top_k=pool_k, filter_metadata=filter_metadata)
        sparse_hits = self.sparse_retriever.search(query, top_k=pool_k, filter_metadata=filter_metadata)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, EvidenceChunk] = {}

        # Process dense rankings
        for rank, hit in enumerate(dense_hits):
            chunk_map[hit.chunk_id] = hit
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (
                self.dense_weight / (self.rrf_k + rank + 1)
            )

        # Process sparse rankings
        for rank, hit in enumerate(sparse_hits):
            if hit.chunk_id not in chunk_map:
                chunk_map[hit.chunk_id] = hit
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (
                self.sparse_weight / (self.rrf_k + rank + 1)
            )

        if not rrf_scores:
            return []

        # Normalize RRF scores to 0.0 - 1.0 range
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_results: List[EvidenceChunk] = []
        for cid in sorted_chunk_ids[:k]:
            orig = chunk_map[cid]
            norm_score = round(rrf_scores[cid] / max_rrf, 4)
            fused_results.append(
                EvidenceChunk(
                    chunk_id=orig.chunk_id,
                    doc_id=orig.doc_id,
                    content=orig.content,
                    score=norm_score,
                    strategy="hybrid_retrieval",
                    page_number=orig.page_number,
                    metadata=orig.metadata
                )
            )

        return fused_results

