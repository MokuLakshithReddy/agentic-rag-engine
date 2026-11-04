"""
Dense Neural Semantic Retrieval Index.
Uses SentenceTransformers (all-MiniLM-L6-v2) for deep semantic embeddings
and normalized cosine similarity search, with deterministic vector fallback.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.state.state import EvidenceChunk

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class DenseRetriever:
    """
    In-memory dense semantic vector index with neural embeddings.
    """

    def __init__(self, top_k: int = 5, min_score: float = 0.05, model_name: str = "all-MiniLM-L6-v2"):
        self.top_k = top_k
        self.min_score = min_score
        self.model_name = model_name
        self.chunks: List[EvidenceChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._is_indexed = False
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Load model without crashing if offline
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None

    def _compute_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Hashed multi-dimensional dense feature representation."""
        dim = 384
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for w in words:
                idx = hash(w) % dim
                vecs[i, idx] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs

    def build_index(self, chunks: List[EvidenceChunk]) -> None:
        """
        Encodes all evidence chunks into normalized semantic vector representations.
        """
        if not chunks:
            self.chunks = []
            self.embeddings = None
            self._is_indexed = False
            return

        self.chunks = list(chunks)
        corpus = [c.content for c in self.chunks]

        if self._model is not None:
            try:
                self.embeddings = self._model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
            except Exception:
                self.embeddings = self._compute_fallback_embeddings(corpus)
        else:
            self.embeddings = self._compute_fallback_embeddings(corpus)

        self._is_indexed = True

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceChunk]:
        """
        Executes semantic cosine similarity vector search.
        """
        k = top_k or self.top_k
        if not self._is_indexed or self.embeddings is None or not self.chunks:
            return []

        if self._model is not None:
            try:
                q_vec = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            except Exception:
                q_vec = self._compute_fallback_embeddings([query])[0]
        else:
            q_vec = self._compute_fallback_embeddings([query])[0]

        # Cosine similarity for normalized vectors is the dot product
        similarities = np.dot(self.embeddings, q_vec).flatten()

        ranked_indices = np.argsort(-similarities)

        results: List[EvidenceChunk] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < self.min_score:
                continue

            chunk = self.chunks[idx]

            # Apply metadata filter
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
