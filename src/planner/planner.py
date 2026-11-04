"""
Retrieval Planner Module.
Formulates an executable retrieval plan, mapping subqueries to optimal tools,
and determining whether reranking or hybrid fusion is required.
"""

from typing import List, Dict, Any
from src.state.state import QueryClassification, Subquery, AgentState


class RetrievalPlanner:
    """
    Intelligent retrieval planner allocating tool strategies based on query nature.
    Avoids redundant heavy workflows for trivial questions.
    """

    def plan(self, state: AgentState) -> List[str]:
        """
        Populates state.selected_tools and updates assigned_tool for each subquery.
        Returns list of selected tool names.
        """
        selected_tools: List[str] = []

        if state.classification == QueryClassification.SIMPLE_FACTUAL:
            # Simple factual query needs direct hybrid retrieval
            selected_tools = ["hybrid_retrieval"]
            for sq in state.subqueries:
                sq.assigned_tool = "hybrid_retrieval"

        elif state.classification == QueryClassification.COMPARATIVE:
            # Comparative queries benefit from multi-query fan-out followed by reranking
            selected_tools = ["multi_query_retrieval", "rerank_evidence"]
            for sq in state.subqueries:
                sq.assigned_tool = "multi_query_retrieval"

        elif state.classification in {QueryClassification.MULTI_HOP, QueryClassification.MULTI_DOCUMENT}:
            selected_tools = ["multi_query_retrieval", "hybrid_retrieval", "rerank_evidence"]
            for sq in state.subqueries:
                sq.assigned_tool = "multi_query_retrieval"

        elif state.classification in {QueryClassification.OUT_OF_SCOPE, QueryClassification.INSUFFICIENTLY_SPECIFIED}:
            # Minimal retrieval to confirm out-of-scope status
            selected_tools = ["sparse_retrieval"]
            for sq in state.subqueries:
                sq.assigned_tool = "sparse_retrieval"

        else:
            # Fallback
            selected_tools = ["hybrid_retrieval"]
            for sq in state.subqueries:
                sq.assigned_tool = "hybrid_retrieval"

        state.selected_tools = selected_tools
        return selected_tools

