"""
Evaluation Metrics Computation.
Computes Recall@K, Precision@K, MRR, NDCG@K, Faithfulness, Citation Accuracy,
and Abstention Correctness.
"""

from typing import List, Set, Dict, Any, Optional
import numpy as np


class EvaluationMetrics:
    """
    Standardized benchmark evaluation metrics for retrieval and generation.
    Handles unanswerable queries strictly without inflating ranking scores.
    """

    @staticmethod
    def recall_at_k(retrieved_doc_ids: List[str], expected_doc_ids: List[str], k: int = 5) -> Optional[float]:
        """Recall@K evaluates only queries where expected ground-truth documents exist."""
        if not expected_doc_ids:
            return None
        retrieved_set = set(retrieved_doc_ids[:k])
        expected_set = set(expected_doc_ids)
        hits = len(retrieved_set.intersection(expected_set))
        return round(hits / len(expected_set), 4)

    @staticmethod
    def precision_at_k(retrieved_doc_ids: List[str], expected_doc_ids: List[str], k: int = 5) -> Optional[float]:
        """Precision@K evaluates only queries where expected ground-truth documents exist."""
        if not expected_doc_ids:
            return None
        actual_k = min(k, len(retrieved_doc_ids))
        if actual_k == 0:
            return 0.0
        retrieved_set = set(retrieved_doc_ids[:actual_k])
        expected_set = set(expected_doc_ids)
        hits = len(retrieved_set.intersection(expected_set))
        return round(hits / actual_k, 4)

    @staticmethod
    def mean_reciprocal_rank(retrieved_doc_ids: List[str], expected_doc_ids: List[str]) -> Optional[float]:
        """MRR calculates reciprocal rank of the first relevant document. Returns None if unanswerable."""
        if not expected_doc_ids:
            return None
        expected_set = set(expected_doc_ids)
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in expected_set:
                return round(1.0 / rank, 4)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_doc_ids: List[str], expected_doc_ids: List[str], k: int = 5) -> Optional[float]:
        """NDCG@K calculates graded ranking gain against ideal ranking."""
        if not expected_doc_ids:
            return None
        expected_set = set(expected_doc_ids)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids[:k]):
            rel = 1.0 if doc_id in expected_set else 0.0
            dcg += rel / np.log2(i + 2)

        ideal_rels = [1.0] * min(len(expected_set), k)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
        return round(dcg / idcg, 4) if idcg > 0 else 0.0

    @staticmethod
    def abstention_correctness(is_answerable: bool, response_status: str) -> float:
        """
        Evaluates abstention reliability:
        - 1.0 if system abstains when unanswerable, or answers when answerable.
        - 0.0 if system hallucinates on unanswerable query or wrongly abstains on answerable query.
        """
        is_abstained = "ABSTAINED" in response_status
        if not is_answerable:
            return 1.0 if is_abstained else 0.0
        else:
            return 0.0 if is_abstained else 1.0
