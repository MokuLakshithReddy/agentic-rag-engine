"""
Evidence Evaluator and Grading Module.
Evaluates retrieved evidence against original query and subquestions,
detects coverage gaps, and assigns calibrated evidence grades.
"""

from typing import List, Dict, Any, Tuple
from src.state.state import AgentState, EvidenceChunk, EvidenceGrading


class EvidenceEvaluator:
    """
    Assesses relevance, subquestion coverage, and identifies information gaps.
    Distinguishes between: Relevant, Partially Relevant, Insufficient, Contradictory, No Evidence.
    """

    def __init__(self, relevance_threshold: float = 0.35, min_hits: int = 1):
        self.relevance_threshold = relevance_threshold
        self.min_hits = min_hits

    def evaluate(self, state: AgentState) -> Tuple[EvidenceGrading, float, List[str]]:
        """
        Evaluates state.retrieved_evidence and returns:
        (EvidenceGrading, overall_relevance_score, missing_gaps)
        """
        evidence = state.retrieved_evidence
        if not evidence:
            return EvidenceGrading.NO_EVIDENCE, 0.0, ["No evidence retrieved from index."]

        # Calculate average top relevance score
        top_scores = [c.score for c in evidence[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        # Assess coverage across subqueries
        missing_gaps: List[str] = []
        covered_subqueries = 0

        for sq in state.subqueries:
            sq_tokens = set(sq.query_text.lower().split())
            sq_covered = False
            for chunk in evidence:
                chunk_tokens = set(chunk.content.lower().split())
                overlap = len(sq_tokens.intersection(chunk_tokens))
                # If target entity is in chunk or good overlap
                if sq.target_entity and sq.target_entity.lower() in chunk.content.lower():
                    sq_covered = True
                    break
                if overlap >= min(2, len(sq_tokens)):
                    sq_covered = True
                    break

            if sq_covered:
                covered_subqueries += 1
                sq.resolved = True
            else:
                missing_gaps.append(f"Insufficient information for: '{sq.query_text}'")

        coverage_ratio = covered_subqueries / max(1, len(state.subqueries))

        # Check for potential contradictory statements (e.g. negation contrasts)
        has_contradiction = False
        all_text = " ".join([c.content.lower() for c in evidence])
        if "is true" in all_text and "is false" in all_text:
            has_contradiction = True

        # Assign Grade
        if avg_score < 0.15 or coverage_ratio == 0:
            grade = EvidenceGrading.NO_EVIDENCE
        elif avg_score < self.relevance_threshold or coverage_ratio < 0.5:
            grade = EvidenceGrading.INSUFFICIENT
        elif has_contradiction:
            grade = EvidenceGrading.CONTRADICTORY
        elif coverage_ratio < 1.0:
            grade = EvidenceGrading.PARTIALLY_RELEVANT
        else:
            grade = EvidenceGrading.RELEVANT

        return grade, round(avg_score, 4), missing_gaps

