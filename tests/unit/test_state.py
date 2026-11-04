"""
Unit tests for Agent State and transition models.
"""

import pytest
from src.state.state import (
    AgentState,
    QueryClassification,
    EvidenceGrading,
    ResponseStatus,
    Subquery,
    EvidenceChunk,
    ToolCallRecord
)


def test_agent_state_initialization():
    state = AgentState(original_query="What is the consensus mechanism in Raft?")
    assert state.original_query == "What is the consensus mechanism in Raft?"
    assert state.classification == QueryClassification.SIMPLE_FACTUAL
    assert state.tool_budget_remaining == 6
    assert state.execution_attempt_count == 0
    assert state.response_status == ResponseStatus.SUCCESS
    assert len(state.trace) == 0


def test_state_trace_recording():
    state = AgentState(original_query="Test query")
    state.add_trace("TEST_STAGE", "Trace message 1", {"key": "val"})
    assert len(state.trace) == 1
    assert state.trace[0].step_index == 1
    assert state.trace[0].stage == "TEST_STAGE"
    assert state.trace[0].metadata["key"] == "val"


def test_tool_budget_decrement():
    state = AgentState(original_query="Test budget")
    assert state.tool_budget_remaining == 6
    record = ToolCallRecord(tool_name="hybrid_retrieval", query_input="test")
    state.record_tool_call(record)
    assert state.tool_budget_remaining == 5
    assert len(state.tool_call_history) == 1

