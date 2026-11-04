"""
Autonomous LLM-Driven Query Decomposer.
Dynamically breaks complex, comparative, and multi-hop queries into targeted subqueries
using structured LLM reasoning with deterministic fallbacks.
"""

from typing import List, Optional
import re
from pydantic import BaseModel, Field
from src.state.state import QueryClassification, Subquery
from src.llm.base import BaseLLMClient, LLMMessage
from src.llm.client import get_llm_client


class SubqueryPlan(BaseModel):
    query_text: str = Field(..., description="Targeted search query")
    target_entity: Optional[str] = Field(None, description="Primary entity or topic being investigated")
    rationale: str = Field(default="", description="Why this subquery is necessary")


class DecompositionOutput(BaseModel):
    subqueries: List[SubqueryPlan] = Field(default_factory=list)
    strategy_explanation: str = ""


DECOMPOSER_SYSTEM_PROMPT = """You are an Autonomous Query Decomposer for an Agentic RAG research engine.
Your task is to analyze user queries and produce targeted, orthogonal subqueries that maximize retrieval precision.

GUIDELINES:
1. For simple questions, return exactly 1 direct subquery.
2. For comparative questions (e.g. A vs B), produce:
   - One targeted query for entity A's architecture/specifications.
   - One targeted query for entity B's architecture/specifications.
   - One comparative query focusing on the differences/trade-offs.
3. For multi-hop questions, identify the prerequisite facts and generate independent queries for each step.
4. Output strictly valid JSON matching the schema.
"""


class QueryDecomposer:
    """
    LLM-powered query decomposer producing adaptive subquery execution graphs.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm = llm_client or get_llm_client()

    def _fallback_decompose(
        self,
        query: str,
        classification: QueryClassification,
        topics: List[str]
    ) -> List[Subquery]:
        """Deterministic heuristic fallback when LLM is offline."""
        q = query.strip()
        if classification == QueryClassification.COMPARATIVE and len(topics) >= 2:
            return [
                Subquery(query_text=f"{topics[0]} architecture and features", target_entity=topics[0]),
                Subquery(query_text=f"{topics[1]} architecture and features", target_entity=topics[1]),
                Subquery(query_text=f"differences and trade-offs between {topics[0]} and {topics[1]}", target_entity=f"{topics[0]}_{topics[1]}")
            ]
        return [Subquery(query_text=q, target_entity=topics[0] if topics else None)]

    def decompose(
        self,
        query: str,
        classification: QueryClassification,
        topics: List[str]
    ) -> List[Subquery]:
        """
        Decomposes query using LLM structured generation with fallback.
        """
        if classification in {QueryClassification.SIMPLE_FACTUAL, QueryClassification.OUT_OF_SCOPE, QueryClassification.INSUFFICIENTLY_SPECIFIED}:
            return [Subquery(query_text=query.strip(), target_entity=topics[0] if topics else None)]

        try:
            messages = [
                LLMMessage(role="system", content=DECOMPOSER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=f"Decompose this {classification.value} query:\n'{query}'\nDetected topics: {topics}")
            ]
            plan = self.llm.generate_structured(messages, DecompositionOutput)
            if plan.subqueries:
                return [
                    Subquery(
                        query_text=sq.query_text,
                        target_entity=sq.target_entity,
                        assigned_tool="hybrid_retrieval"
                    )
                    for sq in plan.subqueries[:4]  # Enforce subquery budget cap
                ]
            return self._fallback_decompose(query, classification, topics)
        except Exception:
            return self._fallback_decompose(query, classification, topics)
