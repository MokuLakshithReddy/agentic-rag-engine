"""
Query Understanding and Intent Classification Module.
Classifies user queries into discrete operational categories and extracts
core conceptual entities.
"""

from typing import Dict, Any, List, Tuple
import re
from src.state.state import QueryClassification


class QueryAnalyzer:
    """
    Deterministic rule-based query classifier with structured heuristics.
    Detects comparative patterns, multi-document indicators, out-of-scope markers,
    and subquery requirements.
    """

    COMPARATIVE_PATTERNS = [
        r"\bcompare\b", r"\bdifference\b", r"\bdifferences\b", r"\bversus\b",
        r"\bvs\b", r"\bbetter than\b", r"\btrade-?offs?\b", r"\bin contrast\b"
    ]

    MULTI_HOP_PATTERNS = [
        r"\bhow does .* lead to\b", r"\bwhat causes .* and how\b",
        r"\bafter .* what happens to\b", r"\bwhy did .* result in\b"
    ]

    OUT_OF_SCOPE_PATTERNS = [
        r"\bweather in\b", r"\bcurrent stock price of\b", r"\btell me a joke\b",
        r"\bwrite a poem\b", r"\bwho won yesterday\b", r"\bplay music\b"
    ]

    def normalize_query(self, query: str) -> str:
        """Cleans whitespace, removes non-printable characters."""
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", query)
        return " ".join(cleaned.strip().split())

    def extract_topics_and_entities(self, query: str) -> List[str]:
        """Extracts significant capitalized phrases or noun phrases."""
        # Clean query
        q = self.normalize_query(query)
        words = q.split()
        stopwords = {
            "what", "is", "the", "difference", "between", "and", "or", "how", "do",
            "does", "in", "of", "to", "a", "an", "compare", "explain", "for", "with"
        }
        entities = [w.strip("?,.:;\"'") for w in words if w.lower() not in stopwords and len(w) > 2]
        return list(dict.fromkeys(entities))  # Deduplicate while preserving order

    def analyze(self, query: str) -> Tuple[QueryClassification, List[str]]:
        """
        Analyzes and classifies the query.
        """
        norm_q = self.normalize_query(query)
        lower_q = norm_q.lower()

        # Check for out-of-scope
        for p in self.OUT_OF_SCOPE_PATTERNS:
            if re.search(p, lower_q):
                return QueryClassification.OUT_OF_SCOPE, []

        # Check for empty / insufficiently specified
        if len(norm_q) < 5 or len(norm_q.split()) <= 1:
            return QueryClassification.INSUFFICIENTLY_SPECIFIED, []

        # Check for comparative
        for p in self.COMPARATIVE_PATTERNS:
            if re.search(p, lower_q):
                topics = self.extract_topics_and_entities(norm_q)
                return QueryClassification.COMPARATIVE, topics

        # Check for multi-hop
        for p in self.MULTI_HOP_PATTERNS:
            if re.search(p, lower_q):
                topics = self.extract_topics_and_entities(norm_q)
                return QueryClassification.MULTI_HOP, topics

        # Check for multi-document markers
        if "both" in lower_q or "across" in lower_q or "all documents" in lower_q or "each approach" in lower_q:
            topics = self.extract_topics_and_entities(norm_q)
            return QueryClassification.MULTI_DOCUMENT, topics

        # Check for ambiguous
        if lower_q.startswith("why?") or lower_q.startswith("what?") or lower_q == "help":
            return QueryClassification.AMBIGUOUS, []

        # Default to simple factual
        topics = self.extract_topics_and_entities(norm_q)
        return QueryClassification.SIMPLE_FACTUAL, topics

