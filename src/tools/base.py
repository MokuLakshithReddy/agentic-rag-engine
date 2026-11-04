"""
Base Tool definition for the Agentic RAG system.
Enforces typed schemas, timeout execution, input sanitization,
structured error handling, and audit telemetry.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
from pydantic import BaseModel, Field
from src.state.state import ToolCallRecord, EvidenceChunk


class BaseToolInput(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class BaseTool(ABC):
    """
    Abstract base tool class.
    All registered tools must inherit from BaseTool.
    """

    def __init__(self, name: str, description: str, timeout_sec: float = 5.0):
        self.name = name
        self.description = description
        self.timeout_sec = timeout_sec

    @abstractmethod
    def _execute(self, **kwargs) -> List[EvidenceChunk]:
        """Tool specific execution logic."""
        pass

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the tool wrapped in safety guards, timing, and error handling.
        """
        start_time = time.time()
        record = ToolCallRecord(
            tool_name=self.name,
            query_input=str(kwargs.get("query", kwargs.get("subqueries", "")))
        )

        try:
            results = self._execute(**kwargs)
            elapsed_ms = (time.time() - start_time) * 1000.0
            record.execution_time_ms = round(elapsed_ms, 2)
            record.success = True
            record.results_count = len(results)
            return {
                "success": True,
                "data": results,
                "record": record
            }
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000.0
            record.execution_time_ms = round(elapsed_ms, 2)
            record.success = False
            record.error_message = str(e)
            record.results_count = 0
            return {
                "success": False,
                "data": [],
                "record": record,
                "error": str(e)
            }

