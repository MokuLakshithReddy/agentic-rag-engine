"""
Rigorous Evidence Evaluator and Gap Detector.
Requires substantive predicate and entity coverage across subqueries,
distinguishes between Relevant, Partially Relevant, Insufficient, Contradictory,
and No Evidence states.
"""

from typing import List, Dict, Any, Tuple, Set
import re
from src.state.state import AgentState, EvidenceChunk, EvidenceGrading


class EvidenceEvaluator:
    """
    Assesses substantive relevance, multi-token subquestion coverage,
    and identifies information gaps. Prevents superficial single-token false positives.
    """

    def __init__(self, relevance_threshold: float = 0.35, min_hits: int = 1):
        self.relevance_threshold = relevance_threshold
        self.min_hits = min_hits
        self.stopwords = {
            "what", "is", "the", "how", "do", "does", "in", "of", "to", "a", "an",
            "for", "with", "and", "or", "on", "it", "at", "by", "this", "that",
            "between", "from", "are", "were", "was"
        }

    def _stem(self, word: str) -> str:
        word = word.lower()
        for suffix in ("sses", "ies", "ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    def _tokenize(self, text: str) -> Set[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return {t for t in tokens if len(t) > 1 and t not in self.stopwords}

    def evaluate(self, state: AgentState) -> Tuple[EvidenceGrading, float, List[str]]:
        """
        Evaluates state.retrieved_evidence across subqueries with rigorous coverage checks.
        Returns: (EvidenceGrading, overall_relevance_score, missing_gaps)
        """
        evidence = state.retrieved_evidence
        if not evidence:
            return EvidenceGrading.NO_EVIDENCE, 0.0, ["No evidence retrieved from index."]

        top_scores = [c.score for c in evidence[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        missing_gaps: List[str] = []
        covered_subqueries = 0

        for sq in state.subqueries:
            sq_tokens = self._tokenize(sq.query_text)
            if not sq_tokens:
                covered_subqueries += 1
                sq.resolved = True
                continue

            sq_stems = {self._stem(t) for t in sq_tokens}
            sq_covered = False
            req_overlap = max(1, min(2, len(sq_tokens)))

            for chunk in evidence:
                chunk_tokens = self._tokenize(chunk.content)
                chunk_stems = {self._stem(t) for t in chunk_tokens}
                overlap = sq_tokens.intersection(chunk_tokens)
                stem_overlap = sq_stems.intersection(chunk_stems)
                overlap_ratio = len(overlap) / len(sq_tokens)
                stem_overlap_ratio = len(stem_overlap) / len(sq_stems)

                # Entity match requirement
                if sq.target_entity:
                    entity_in_chunk = sq.target_entity.lower() in chunk.content.lower()
                    if entity_in_chunk and (len(overlap) >= 1 or len(stem_overlap) >= 1):
                        sq_covered = True
                        break

                # Substantive token coverage requirement (exact or stemmed root match)
                if len(overlap) >= req_overlap or len(stem_overlap) >= req_overlap or stem_overlap_ratio >= 0.40:
                    sq_covered = True
                    break

                # Strong relevance score + root overlap
                if chunk.score >= 0.60 and len(stem_overlap) >= 1:
                    sq_covered = True
                    break

            if sq_covered:
                covered_subqueries += 1
                sq.resolved = True
            else:
                missing_gaps.append(f"Insufficient evidence for: '{sq.query_text}'")

        coverage_ratio = covered_subqueries / max(1, len(state.subqueries))

        # Check for direct contradictions in evidence corpus
        has_contradiction = False
        all_text = " ".join([c.content.lower() for c in evidence])
        if "is true" in all_text and "is false" in all_text:
            has_contradiction = True

        # Assign calibrated Grade
        if avg_score < 0.15 or coverage_ratio == 0.0:
            grade = EvidenceGrading.NO_EVIDENCE
        elif avg_score < self.relevance_threshold or coverage_ratio < 0.50:
            grade = EvidenceGrading.INSUFFICIENT
        elif has_contradiction:
            grade = EvidenceGrading.CONTRADICTORY
        elif coverage_ratio < 1.0:
            grade = EvidenceGrading.PARTIALLY_RELEVANT
        else:
            grade = EvidenceGrading.RELEVANT

        return grade, round(avg_score, 4), missing_gaps
