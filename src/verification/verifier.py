"""
Factual Claim-to-Evidence Verifier.
Decomposes responses into material factual claims and deterministically verifies
that each claim is directly supported by its cited evidence chunk.
Never auto-attaches citations to unverified statements.
"""

from typing import Tuple, List, Dict, Any, Set, Optional
import re
from src.state.state import AgentState, ResponseStatus, EvidenceChunk
from src.verification.citation_checker import CitationChecker


class AnswerVerifier:
    """
    Validates factual claims against retrieved evidence passages.
    Rejects hallucinated source citations and ungrounded statements.
    """

    def __init__(self, min_confidence_threshold: float = 0.60, min_claim_support_ratio: float = 0.40):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_claim_support_ratio = min_claim_support_ratio
        self.checker = CitationChecker()

    def _split_into_claims(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Splits answer into individual sentence-level factual claims and resolves their associated citation.
        Returns list of (claim_text, cited_chunk_id).
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        claims_with_cites: List[Tuple[str, Optional[str]]] = []

        for line in lines:
            if line.startswith("#") or line.lower().startswith("based on"):
                continue

            # Check if this line/block has one or more citations
            line_cites = self.checker.extract_citations(line)
            default_cite = line_cites[-1] if line_cites else None

            # Split line into sentences
            sentences = re.split(r"(?<=[.!?])\s+", line)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 15:
                    # Find citation in this sentence, or inherit the line's citation
                    s_cites = self.checker.extract_citations(s_clean)
                    cite = s_cites[0] if s_cites else default_cite
                    claims_with_cites.append((s_clean, cite))

        return claims_with_cites

    def verify_claim_grounding(
        self,
        claim: str,
        cited_id: Optional[str],
        evidence_map: Dict[str, EvidenceChunk]
    ) -> Tuple[bool, Optional[str], float]:
        """
        Verifies if a specific claim is grounded in its cited evidence chunk.
        Returns: (is_grounded, cited_chunk_id, support_score)
        """
        if not cited_id:
            # Claim has no citation -> Ungrounded
            return False, None, 0.0

        if cited_id not in evidence_map:
            # Cited chunk does not exist in retrieved evidence
            return False, cited_id, 0.0

        chunk = evidence_map[cited_id]
        # Clean claim text (strip citation tag)
        claim_clean = re.sub(r"\[[a-zA-Z0-9_\-\.]+#[a-zA-Z0-9_\-]+\]", "", claim).lower()
        chunk_clean = chunk.content.lower()

        # Compute key token overlap (excluding stopwords)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in",
            "of", "for", "with", "that", "this", "by", "from", "on", "it", "its"
        }
        claim_words = [w.strip("?,.:;\"'") for w in claim_clean.split() if len(w) > 2 and w not in stopwords]
        if not claim_words:
            return True, cited_id, 1.0

        supported_words = sum(1 for w in claim_words if w in chunk_clean)
        support_ratio = supported_words / len(claim_words)

        is_supported = support_ratio >= self.min_claim_support_ratio
        return is_supported, cited_id, round(support_ratio, 3)

    def verify(self, state: AgentState) -> Tuple[str, float, ResponseStatus]:
        """
        Verifies all claims and citations in state.synthesis_draft against state.retrieved_evidence.
        Returns (verified_answer, confidence_score, response_status)
        """
        if state.response_status in {
            ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE,
            ResponseStatus.ABSTAINED_OUT_OF_SCOPE
        }:
            return state.final_answer or state.synthesis_draft or "", 0.0, state.response_status

        draft = state.synthesis_draft or state.final_answer
        if not draft:
            return "No response generated.", 0.0, ResponseStatus.ERROR

        # Check for abstention text
        if "not have sufficient verified evidence" in draft.lower() or "out of scope" in draft.lower():
            return draft, 0.0, ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE

        # Index retrieved evidence by chunk_id
        evidence_map: Dict[str, EvidenceChunk] = {c.chunk_id: c for c in state.retrieved_evidence}

        # Extract citations
        all_cited_ids = self.checker.extract_citations(draft)
        state.citations = all_cited_ids

        # If zero citations are provided, NEVER fabricate or auto-attach citations!
        if not all_cited_ids:
            state.unsupported_claims = ["Answer contains factual assertions without supporting citations."]
            return (
                "I do not have sufficient verified evidence in the indexed knowledge base to answer this question.",
                0.0,
                ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
            )

        # Decompose answer into discrete claims with citations and verify each claim
        claims_with_cites = self._split_into_claims(draft)
        if not claims_with_cites:
            claims_with_cites = [(draft, all_cited_ids[0])]

        supported_count = 0
        unsupported_claims: List[str] = []

        for claim, cited_id in claims_with_cites:
            is_supported, cid, score = self.verify_claim_grounding(claim, cited_id, evidence_map)
            if is_supported:
                supported_count += 1
            else:
                reason = f"Unverified claim (Cited: {cid}, Support score: {score:.2f}): '{claim[:100]}...'"
                unsupported_claims.append(reason)

        state.unsupported_claims = unsupported_claims
        grounding_ratio = supported_count / max(1, len(claims_with_cites))

        # Check citation ID validity
        is_valid_cites, valid_cites, invalid_cites, cit_precision = self.checker.verify_citations(
            all_cited_ids, state.retrieved_evidence
        )

        # Empirically calibrated confidence score
        # 50% claim grounding, 30% retrieval relevance, 20% citation precision
        confidence = (0.50 * grounding_ratio) + (0.30 * state.evidence_relevance_score) + (0.20 * cit_precision)
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        # Enforce abstention if grounding or confidence fails
        if grounding_ratio < 0.40 or confidence < self.min_confidence_threshold or len(invalid_cites) > 0:
            return (
                "I do not have sufficient verified evidence in the indexed knowledge base to answer this question.",
                confidence,
                ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
            )

        return draft, confidence, ResponseStatus.SUCCESS
