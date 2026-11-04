"""
Citation Verification Module.
Extracts citations from generated text, verifies their existence in retrieved
evidence, and validates that cited chunks match ground truth document IDs.
"""

from typing import List, Dict, Any, Set, Tuple
import re
from src.state.state import EvidenceChunk


class CitationChecker:
    """
    Deterministic citation audit engine.
    Ensures citations refer to real, retrieved chunk identifiers.
    """

    CITATION_PATTERN = r"\[([a-zA-Z0-9_\-\.]+#[a-zA-Z0-9_\-]+)\]"

    def extract_citations(self, text: str) -> List[str]:
        """Extracts all citation references matching [doc_id#chunk_id]."""
        matches = re.findall(self.CITATION_PATTERN, text)
        return list(dict.fromkeys(matches))

    def verify_citations(
        self,
        citations: List[str],
        retrieved_evidence: List[EvidenceChunk]
    ) -> Tuple[bool, List[str], List[str], float]:
        """
        Verifies citations against retrieved evidence index.
        Returns:
        (is_valid, valid_citations, invalid_citations, precision_score)
        """
        valid_ids: Set[str] = {c.chunk_id for c in retrieved_evidence}

        if not citations:
            return False, [], [], 0.0

        valid_citations: List[str] = []
        invalid_citations: List[str] = []

        for cite in citations:
            if cite in valid_ids:
                valid_citations.append(cite)
            else:
                invalid_citations.append(cite)

        precision = len(valid_citations) / len(citations) if citations else 0.0
        is_valid = len(invalid_citations) == 0 and len(valid_citations) > 0

        return is_valid, valid_citations, invalid_citations, round(precision, 4)

