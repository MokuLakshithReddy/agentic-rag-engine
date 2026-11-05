"""
Sparse lexical retrieval index.
Uses BM25Okapi algorithm for exact keyword matching, acronym resolution,
and term-frequency ranking.
"""

from typing import List, Dict, Any, Optional
import re
from rank_bm25 import BM25Okapi
from src.state.state import EvidenceChunk


class SparseRetriever:
    """
    BM25 sparse keyword retriever with tokenization and score normalization.
    """

    def __init__(self, top_k: int = 5, min_score: float = 0.01):
        self.top_k = top_k
        self.min_score = min_score
        self.chunks: List[EvidenceChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self._is_indexed = False

    def _tokenize(self, text: str) -> List[str]:
        # Clean and split into lowercased alphanumeric tokens
        return re.findall(r"\b\w+\b", text.lower())

    def build_index(self, chunks: List[EvidenceChunk]) -> None:
        """
        Builds BM25 inverted index from evidence chunks.
        """
        if not chunks:
            self.chunks = []
            self.bm25 = None
            self._is_indexed = False
            return

        self.chunks = list(chunks)
        tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._is_indexed = True

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[Any] = None
    ) -> List[EvidenceChunk]:
        """
        Executes BM25 keyword search with cooperative cancellation check.
        """
        if cancel_token and getattr(cancel_token, "is_set", lambda: False)():
            return []

        k = top_k or self.top_k
        if not self._is_indexed or self.bm25 is None or not self.chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        raw_scores = self.bm25.get_scores(tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0

        # Sort indices by descending score
        ranked_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)

        results: List[EvidenceChunk] = []
        for idx in ranked_indices:
            if cancel_token and getattr(cancel_token, "is_set", lambda: False)():
                break
            raw_s = float(raw_scores[idx])
            if raw_s <= 0.0:
                continue
            norm_score = round(raw_s / max_score, 4)
            if norm_score < self.min_score:
                continue

            chunk = self.chunks[idx]

            # Apply metadata filtering if specified
            if filter_metadata:
                match = True
                for fk, fv in filter_metadata.items():
                    if chunk.metadata.get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue

            result_chunk = EvidenceChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                score=norm_score,
                strategy="sparse_retrieval",
                page_number=chunk.page_number,
                metadata=chunk.metadata
            )
            results.append(result_chunk)
            if len(results) >= k:
                break

        return results

