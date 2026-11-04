"""
Tests for explicit abstention under out-of-scope and unanswerable queries.
"""

from src.ingestion.indexer import KnowledgeBase
from src.tools.registry import ToolRegistry
from src.agents.workflow import AgenticRAGWorkflow
from src.state.state import ResponseStatus


def test_abstention_out_of_scope():
    kb = KnowledgeBase()
    registry = ToolRegistry(kb)
    workflow = AgenticRAGWorkflow(registry)

    state = workflow.run("What is the current stock price of Apple?")
    assert state.response_status == ResponseStatus.ABSTAINED_OUT_OF_SCOPE
    assert "out of scope" in state.final_answer.lower()
    assert len(state.citations) == 0


def test_abstention_missing_evidence():
    kb = KnowledgeBase()
    kb.index_document("doc_crypto", "Bitcoin uses Proof of Work consensus.")
    registry = ToolRegistry(kb)
    workflow = AgenticRAGWorkflow(registry)

    state = workflow.run("What is the chemical composition of lunar regolith?")
    assert state.response_status == ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
    assert "not have sufficient verified evidence" in state.final_answer.lower()

