"""
Unit tests for Tool Registry, Sandboxing, Capacity Limits, and Process Isolation.
"""

import pytest
import time
from src.ingestion.indexer import KnowledgeBase
from src.tools.registry import ToolRegistry
from src.tools.base import BaseTool, execute_with_process_isolation
from src.state.state import EvidenceChunk


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


def test_tool_capacity_limit_rejection():
    """When worker pool and queue capacity is saturated, tasks must be rejected immediately."""
    class DummyTool(BaseTool):
        def _execute(self, **kwargs):
            return []

    dummy = DummyTool(name="dummy", description="dummy")
    # Simulate saturated capacity
    original = BaseTool._active_tasks
    try:
        BaseTool._active_tasks = BaseTool.MAX_CONCURRENT_WORKERS + BaseTool.MAX_PENDING_QUEUE
        res = dummy.run(query="overflow")
        assert res["success"] is False
        assert "capacity exceeded" in res["error"].lower()
    finally:
        BaseTool._active_tasks = original


def test_process_isolated_execution():
    """Verifies process-isolated worker execution."""
    res = execute_with_process_isolation(abs, (-42,), timeout_sec=2.0)
    assert res["success"] is True
    assert res["data"] == 42
