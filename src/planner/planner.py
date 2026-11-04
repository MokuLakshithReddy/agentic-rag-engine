"""
Adaptive Retrieval Planner Module.
Uses LLM reasoning to dynamically allocate retrieval tools across subqueries,
deciding when dense, sparse, hybrid, or metadata search is optimal.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.state.state import QueryClassification, Subquery, AgentState
from src.llm.base import BaseLLMClient, LLMMessage
from src.llm.client import get_llm_client


class ToolSelectionPlan(BaseModel):
    selected_tools: List[str] = Field(..., description="Ordered list of tools to invoke")
    subquery_tools: Dict[str, str] = Field(default_factory=dict, description="Map of subquery index to tool name")
    rationale: str = Field(default="", description="Strategic justification for tool choices")


PLANNER_SYSTEM_PROMPT = """You are an Adaptive Retrieval Planner for an Agentic RAG engine.
Given a user question, its classification, and decomposed subqueries, decide which registered tools to execute.

REGISTERED TOOLS:
- dense_retrieval: Semantic embeddings vector search. Best for conceptual, thematic, or abstract queries.
- sparse_retrieval: BM25 keyword search. Best for exact token matches, acronyms, IDs, or technical formulas.
- hybrid_retrieval: Fused dense + sparse search via RRF. Best general-purpose default.
- metadata_search: Faceted search by doc_id, category, or source file.
- multi_query_retrieval: Concurrent fan-out search across multiple subqueries.
- rerank_evidence: Cross-encoder reordering of candidate passages.

Output strictly valid JSON matching the schema.
"""


class RetrievalPlanner:
    """
    Autonomous retrieval planner allocating tool strategies through LLM reasoning.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm = llm_client or get_llm_client()

    def _fallback_plan(self, state: AgentState) -> List[str]:
        """Deterministic heuristic fallback."""
        if state.classification == QueryClassification.COMPARATIVE:
            selected_tools = ["multi_query_retrieval", "rerank_evidence"]
            for sq in state.subqueries:
                sq.assigned_tool = "multi_query_retrieval"
        elif state.classification in {QueryClassification.MULTI_HOP, QueryClassification.MULTI_DOCUMENT}:
            selected_tools = ["multi_query_retrieval", "hybrid_retrieval", "rerank_evidence"]
            for sq in state.subqueries:
                sq.assigned_tool = "multi_query_retrieval"
        elif state.classification in {QueryClassification.OUT_OF_SCOPE, QueryClassification.INSUFFICIENTLY_SPECIFIED}:
            selected_tools = ["sparse_retrieval"]
            for sq in state.subqueries:
                sq.assigned_tool = "sparse_retrieval"
        else:
            selected_tools = ["hybrid_retrieval"]
            for sq in state.subqueries:
                sq.assigned_tool = "hybrid_retrieval"

        state.selected_tools = selected_tools
        return selected_tools

    def plan(self, state: AgentState) -> List[str]:
        """
        Formulates an adaptive retrieval plan using structured LLM reasoning.
        """
        sq_descriptions = "\n".join([f"[{i}] {sq.query_text} (Target: {sq.target_entity})" for i, sq in enumerate(state.subqueries)])
        prompt = (
            f"User Question: '{state.normalized_query}'\n"
            f"Classification: {state.classification.value}\n"
            f"Subqueries:\n{sq_descriptions}"
        )

        try:
            messages = [
                LLMMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt)
            ]
            plan = self.llm.generate_structured(messages, ToolSelectionPlan)
            if plan.selected_tools:
                state.selected_tools = [t for t in plan.selected_tools if t in {
                    "dense_retrieval", "sparse_retrieval", "hybrid_retrieval",
                    "metadata_search", "multi_query_retrieval", "rerank_evidence"
                }]
                # Assign tools to subqueries
                for i, sq in enumerate(state.subqueries):
                    assigned = plan.subquery_tools.get(str(i), state.selected_tools[0] if state.selected_tools else "hybrid_retrieval")
                    sq.assigned_tool = assigned
                return state.selected_tools
            return self._fallback_plan(state)
        except Exception:
            return self._fallback_plan(state)
