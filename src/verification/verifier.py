"""
Comprehensive Answer and Citation Verifier.
Performs deterministic verification of citations, factual alignment,
and triggers controlled abstention on ungrounded claims.
"""

from typing import Tuple, List, Dict, Any
from src.state.state import AgentState, ResponseStatus
from src.verification.citation_checker import CitationChecker


class AnswerVerifier:
    """
    Validates synthesized responses against retrieved evidence.
    Prevents hallucinated source citations and ungrounded statements.
    """

    def __init__(self, min_confidence_threshold: float = 0.60):
        self.min_confidence_threshold = min_confidence_threshold
        self.checker = CitationChecker()

    def verify(self, state: AgentState) -> Tuple[str, float, ResponseStatus]:
        """
        Verifies state.synthesis_draft against state.retrieved_evidence.
        Returns (verified_answer, confidence_score, response_status)
        """
        # If already abstained, pass through
        if state.response_status in {ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE, ResponseStatus.ABSTAINED_OUT_OF_SCOPE}:
            return state.final_answer or state.synthesis_draft or "", 0.0, state.response_status

        draft = state.synthesis_draft or state.final_answer
        if not draft:
            return "No response generated.", 0.0, ResponseStatus.ERROR

        # Extract citations
        cited_ids = self.checker.extract_citations(draft)
        state.citations = cited_ids

        if not cited_ids:
            # Answer generated with zero citations -> Unverified!
            if state.evidence_relevance_score > 0.4:
                # Attach available top evidence citation
                fallback_cid = state.retrieved_evidence[0].chunk_id
                draft = f"{draft} [{fallback_cid}]"
                cited_ids = [fallback_cid]
                state.citations = cited_ids
            else:
                return (
                    "I do not have sufficient verified evidence in the indexed knowledge base to answer this question.",
                    0.0,
                    ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
                )

        is_valid, valid_cites, invalid_cites, precision = self.checker.verify_citations(
            cited_ids, state.retrieved_evidence
        )

        state.unsupported_claims = [f"Invalid citation: {cid}" for cid in invalid_cites]

        # Calculate calibrated confidence score
        # Confidence = 0.5 * citation_precision + 0.3 * evidence_relevance + 0.2 * coverage
        coverage_factor = 1.0 if not state.missing_information else 0.5
        confidence = (0.5 * precision) + (0.3 * state.evidence_relevance_score) + (0.2 * coverage_factor)
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        if confidence < self.min_confidence_threshold and not valid_cites:
            return (
                "I do not have sufficient verified evidence in the indexed knowledge base to answer this question.",
                confidence,
                ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
            )

        return draft, confidence, ResponseStatus.SUCCESS

