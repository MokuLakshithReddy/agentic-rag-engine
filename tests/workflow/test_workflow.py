"""
Workflow tests for state transitions and bounded loop termination.
"""

from src.ingestion.indexer import KnowledgeBase
from src.tools.registry import ToolRegistry
from src.agents.workflow import AgenticRAGWorkflow
from src.state.state import QueryClassification, ResponseStatus


def test_workflow_state_transitions():
    kb = KnowledgeBase()
    kb.index_document("doc_simple", "B-Tree index maintains balanced multi-way tree nodes for fast lookups.")
    registry = ToolRegistry(kb)
    workflow = AgenticRAGWorkflow(registry)

    state = workflow.run("How do B-Tree indexes work?")
    assert state.classification == QueryClassification.SIMPLE_FACTUAL
    assert state.tool_budget_remaining >= 0
    assert state.execution_attempt_count >= 1
    assert state.response_status == ResponseStatus.SUCCESS


def test_bounded_loop_termination():
    kb = KnowledgeBase()
    # Empty KB should terminate safely without infinite looping
    registry = ToolRegistry(kb)
    workflow = AgenticRAGWorkflow(registry)

    state = workflow.run("A question on an empty index")
    # Must terminate within configured limits
    assert state.planning_revision_count <= state.max_planning_revisions
    assert state.tool_budget_remaining >= 0
    assert "ABSTAINED" in state.response_status.value

