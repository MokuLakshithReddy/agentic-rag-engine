"""
Dense semantic retrieval index.
Provides vector-based semantic retrieval using normalized embeddings and cosine similarity.
Supports offline deterministic vector encoding (TF-IDF + Subword Trigrams) and
optional transformer embedding models.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.state.state import EvidenceChunk


class DenseRetriever:
    """
    In-memory dense semantic vector index with cosine similarity search.
    Guaranteed to run deterministically in restricted/offline environments.
    """

    def __init__(self, top_k: int = 5, min_score: float = 0.05):
        self.top_k = top_k
        self.min_score = min_score
        self.chunks: List[EvidenceChunk] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=50000,
            sublinear_tf=True
        )
        self.embeddings: Optional[np.ndarray] = None
        self._is_indexed = False

    def build_index(self, chunks: List[EvidenceChunk]) -> None:
        """
        Builds the semantic vector matrix for all chunks.
        """
        if not chunks:
            self.chunks = []
            self.embeddings = None
            self._is_indexed = False
            return

        self.chunks = list(chunks)
        corpus = [c.content for c in self.chunks]
        self.embeddings = self.vectorizer.fit_transform(corpus)
        self._is_indexed = True

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceChunk]:
        """
        Executes semantic cosine similarity search.
        """
        k = top_k or self.top_k
        if not self._is_indexed or self.embeddings is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        # Cosine similarity for normalized tf-idf vectors is dot product
        similarities = (self.embeddings * query_vec.T).toarray().flatten()

        # Sort indices by descending score
        ranked_indices = np.argsort(-similarities)

        results: List[EvidenceChunk] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < self.min_score:
                continue

            chunk = self.chunks[idx]

            # Apply metadata filter if provided
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
                score=round(score, 4),
                strategy="dense_retrieval",
                page_number=chunk.page_number,
                metadata=chunk.metadata
            )
            results.append(result_chunk)
            if len(results) >= k:
                break

        return results

