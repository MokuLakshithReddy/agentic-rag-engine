"""
Metadata search and faceted retrieval engine.
Enables exact document ID filtering, category partitioning, source tagging,
and structured metadata constraints.
"""

from typing import List, Dict, Any, Optional
from src.state.state import EvidenceChunk


class MetadataRetriever:
    """
    Faceted search retriever querying structured document metadata.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.chunks: List[EvidenceChunk] = []

    def set_chunks(self, chunks: List[EvidenceChunk]) -> None:
        self.chunks = list(chunks)

    def search(
        self,
        doc_id: Optional[str] = None,
        category: Optional[str] = None,
        source_file: Optional[str] = None,
        custom_filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        cancel_token: Optional[Any] = None
    ) -> List[EvidenceChunk]:
        """
        Filters chunks matching provided metadata criteria with cancellation check.
        """
        if cancel_token and getattr(cancel_token, "is_set", lambda: False)():
            return []

        k = top_k or self.top_k
        results: List[EvidenceChunk] = []

        for chunk in self.chunks:
            if cancel_token and getattr(cancel_token, "is_set", lambda: False)():
                break
            if doc_id and chunk.doc_id.lower() != doc_id.lower():
                continue
            if category and chunk.metadata.get("category", "").lower() != category.lower():
                continue
            if source_file and chunk.metadata.get("source_file", "").lower() != source_file.lower():
                continue
            if custom_filters:
                mismatch = False
                for fk, fv in custom_filters.items():
                    if chunk.metadata.get(fk) != fv:
                        mismatch = True
                        break
                if mismatch:
                    continue

            # Optional lexical match within filtered set
            score = 1.0
            if query:
                q_lower = query.lower()
                c_lower = chunk.content.lower()
                if q_lower not in c_lower:
                    # Partial match
                    tokens = [t for t in q_lower.split() if len(t) > 2]
                    overlap = sum(1 for t in tokens if t in c_lower)
                    if tokens and overlap == 0:
                        continue
                    score = round(overlap / len(tokens), 3) if tokens else 0.5

            results.append(
                EvidenceChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    score=score,
                    strategy="metadata_search",
                    page_number=chunk.page_number,
                    metadata=chunk.metadata
                )
            )
            if len(results) >= k:
                break

        return results

