"""
Unit tests for Tool Registry and Sandboxing.
"""

import pytest
from src.ingestion.indexer import KnowledgeBase
from src.tools.registry import ToolRegistry


def test_tool_registry_registration():
    kb = KnowledgeBase()
    registry = ToolRegistry(kb)
    tools = registry.list_tools()
    tool_names = [t["name"] for t in tools]
    assert "dense_retrieval" in tool_names
    assert "sparse_retrieval" in tool_names
    assert "hybrid_retrieval" in tool_names
    assert "metadata_search" in tool_names
    assert "multi_query_retrieval" in tool_names
    assert "rerank_evidence" in tool_names


def test_unauthorized_tool_execution():
    kb = KnowledgeBase()
    registry = ToolRegistry(kb)
    result = registry.execute_tool("arbitrary_bash_command", command="rm -rf /")
    assert result["success"] is False
    assert "not registered" in result["error"]


def test_hard_tool_timeout_enforcement():
    import time
    from src.tools.base import BaseTool
    from src.state.state import EvidenceChunk

    class SlowTool(BaseTool):
        def __init__(self):
            super().__init__(name="slow_tool", description="Test slow tool", timeout_sec=0.2)

        def _execute(self, **kwargs):
            time.sleep(1.0)
            return [EvidenceChunk(chunk_id="slow#1", doc_id="slow", content="late")]

    slow = SlowTool()
    start = time.time()
    res = slow.run(query="test")
    elapsed = time.time() - start

    assert res["success"] is False
    assert "timed out" in res["error"]
    assert elapsed < 0.6  # Must return immediately, without waiting for the 1.0s sleep

