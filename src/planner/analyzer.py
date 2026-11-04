"""
Query Understanding and Intent Classification Module.
Uses configurable LLM reasoning with structured Pydantic schema parsing,
with deterministic heuristics as fallback.
"""

from typing import Dict, Any, List, Tuple, Optional
import re
from pydantic import BaseModel, Field
from src.state.state import QueryClassification
from src.llm.base import BaseLLMClient, LLMMessage
from src.llm.client import get_llm_client


class QueryAnalysisOutput(BaseModel):
    classification: QueryClassification = QueryClassification.SIMPLE_FACTUAL
    reasoning: str = ""
    topics: List[str] = Field(default_factory=list)


ANALYZER_SYSTEM_PROMPT = """You are an expert Query Analyzer for an Agentic RAG system.
Your job is to analyze the user's research query and classify it into one of:
- simple_factual: Direct question about a single entity, mechanism, or fact.
- multi_document: Requires synthesizing information from multiple distinct documents.
- comparative: Compares two or more architectures, mechanisms, or systems (e.g. versus, differences).
- multi_hop: Requires sequential reasoning steps (A leads to B leads to C).
- ambiguous: Vague, unclear, or underspecified question.
- out_of_scope: Asks about topics unrelated to technical documentation or completely out of domain (e.g., weather, stocks).
- insufficiently_specified: Too short to determine intent.

Output strictly valid JSON with keys: classification, reasoning, topics.
"""


class QueryAnalyzer:
    """
    LLM-driven query intent classifier with structured JSON schema outputs
    and deterministic fallback heuristics.
    """

    COMPARATIVE_PATTERNS = [
        r"\bcompare\b", r"\bdifference\b", r"\bdifferences\b", r"\bversus\b",
        r"\bvs\b", r"\bbetter than\b", r"\btrade-?offs?\b", r"\bin contrast\b"
    ]

    OUT_OF_SCOPE_PATTERNS = [
        r"\bweather in\b", r"\bcurrent stock price\b", r"\btell me a joke\b",
        r"\bwrite a poem\b", r"\bquantum error correction\b"
    ]

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm = llm_client or get_llm_client()

    def normalize_query(self, query: str) -> str:
        """Cleans whitespace, removes non-printable characters."""
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", query)
        return " ".join(cleaned.strip().split())

    def _fallback_analyze(self, query: str) -> Tuple[QueryClassification, List[str]]:
        """Deterministic heuristic fallback when LLM is unavailable."""
        norm_q = self.normalize_query(query)
        lower_q = norm_q.lower()

        for p in self.OUT_OF_SCOPE_PATTERNS:
            if re.search(p, lower_q):
                return QueryClassification.OUT_OF_SCOPE, []

        if len(norm_q) < 5 or len(norm_q.split()) <= 1:
            return QueryClassification.INSUFFICIENTLY_SPECIFIED, []

        for p in self.COMPARATIVE_PATTERNS:
            if re.search(p, lower_q):
                topics = [w.strip("?,.:;\"'") for w in norm_q.split() if len(w) > 3 and w.lower() not in {"compare", "versus", "between", "what", "difference"}]
                return QueryClassification.COMPARATIVE, topics[:3]

        topics = [w.strip("?,.:;\"'") for w in norm_q.split() if len(w) > 3][:3]
        return QueryClassification.SIMPLE_FACTUAL, topics

    def analyze(self, query: str) -> Tuple[QueryClassification, List[str]]:
        """
        Analyzes and classifies query using LLM structured outputs with fallback.
        """
        norm_q = self.normalize_query(query)

        # Quick pre-filter for out-of-scope or empty
        for p in self.OUT_OF_SCOPE_PATTERNS:
            if re.search(p, norm_q.lower()):
                return QueryClassification.OUT_OF_SCOPE, []

        if len(norm_q) < 5 or len(norm_q.split()) <= 1:
            return QueryClassification.INSUFFICIENTLY_SPECIFIED, []

        try:
            messages = [
                LLMMessage(role="system", content=ANALYZER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=f"Analyze this query:\n'{norm_q}'")
            ]
            result = self.llm.generate_structured(messages, QueryAnalysisOutput)
            return result.classification, result.topics
        except Exception:
            return self._fallback_analyze(norm_q)
