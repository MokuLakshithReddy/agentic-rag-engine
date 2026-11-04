"""
Integration tests for End-to-End Agentic RAG Workflow.
"""

from src.ingestion.indexer import KnowledgeBase
from src.tools.registry import ToolRegistry
from src.agents.workflow import AgenticRAGWorkflow
from src.evaluation.dataset import SAMPLE_KNOWLEDGE_CORPUS
from src.state.state import ResponseStatus


def test_end_to_end_agentic_workflow():
    kb = KnowledgeBase()
    for doc in SAMPLE_KNOWLEDGE_CORPUS:
        kb.index_document(doc["doc_id"], doc["text"], {"category": doc["category"]})

    registry = ToolRegistry(kb)
    workflow = AgenticRAGWorkflow(registry)

    # Test comparative query
    state = workflow.run("Compare Ethereum and Solana consensus mechanisms.")
    assert state.response_status == ResponseStatus.SUCCESS
    assert state.confidence_score > 0.5
    assert len(state.citations) > 0
    assert len(state.trace) > 4
    assert len(state.tool_call_history) >= 1

