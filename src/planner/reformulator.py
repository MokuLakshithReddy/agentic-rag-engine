"""
Query Reformulation and Bounded Re-Retrieval Controller.
Implements bounded iteration guards to prevent infinite loops while adapting
retrieval strategies when initial evidence is insufficient.
"""

from typing import List, Dict, Any
from src.state.state import AgentState, Subquery, EvidenceGrading


class QueryReformulator:
    """
    Reformulates queries and updates subqueries when evidence evaluation reveals gaps.
    Enforces maximum revision bounds and tool call budgets.
    """

    def __init__(self, max_revisions: int = 2):
        self.max_revisions = max_revisions

    def can_reformulate(self, state: AgentState) -> bool:
        """
        Returns True if reformulation is permitted within remaining budgets.
        """
        if state.planning_revision_count >= self.max_revisions:
            return False
        if state.tool_budget_remaining <= 1:
            return False
        if state.evidence_grading in {EvidenceGrading.RELEVANT}:
            return False
        return True

    def reformulate(self, state: AgentState) -> List[Subquery]:
        """
        Generates revised subqueries targeting missing gaps.
        Increments revision counter and selects alternative tools.
        """
        state.planning_revision_count += 1
        revised_subqueries: List[Subquery] = []

        # Target unresolved subqueries
        for sq in state.subqueries:
            if not sq.resolved:
                # Expand query or switch strategy
                new_text = f"{sq.query_text} architecture overview explanation"
                # Switch tool from hybrid to sparse or dense if repeated
                alt_tool = "sparse_retrieval" if sq.assigned_tool == "dense_retrieval" else "hybrid_retrieval"
                revised_subqueries.append(
                    Subquery(
                        query_text=new_text,
                        target_entity=sq.target_entity,
                        assigned_tool=alt_tool
                    )
                )

        # If all were resolved but overall evidence was weak, broaden original query
        if not revised_subqueries:
            revised_subqueries.append(
                Subquery(
                    query_text=f"{state.original_query} documentation guide overview",
                    assigned_tool="hybrid_retrieval"
                )
            )

        state.subqueries = revised_subqueries
        return revised_subqueries

