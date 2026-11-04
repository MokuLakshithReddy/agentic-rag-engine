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

