"""
Base Tool definition with genuinely bounded execution timeouts.
Uses a shared, non-blocking persistent ThreadPoolExecutor so that timeout
exceptions return immediately without triggering wait=True shutdown blocking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import concurrent.futures
from pydantic import BaseModel, Field
from src.state.state import ToolCallRecord, EvidenceChunk


class BaseToolInput(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class BaseTool(ABC):
    """
    Abstract base tool class.
    Executes operations under a hard, non-blocking timeout guarantee.
    """

    _shared_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    @classmethod
    def _get_executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        if cls._shared_executor is None:
            cls._shared_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix="rag_tool_worker"
            )
        return cls._shared_executor

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
        Executes the tool with immediate unblocked return upon timeout expiration.
        Avoids the executor context manager shutdown(wait=True) deadlock.
        """
        start_time = time.time()
        record = ToolCallRecord(
            tool_name=self.name,
            query_input=str(kwargs.get("query", kwargs.get("subqueries", "")))[:200]
        )

        executor = self._get_executor()
        future = executor.submit(self._execute, **kwargs)

        try:
            results = future.result(timeout=self.timeout_sec)
            elapsed_ms = (time.time() - start_time) * 1000.0
            record.execution_time_ms = round(elapsed_ms, 2)
            record.success = True
            record.results_count = len(results)
            return {
                "success": True,
                "data": results,
                "record": record
            }
        except concurrent.futures.TimeoutError:
            future.cancel()
            elapsed_ms = (time.time() - start_time) * 1000.0
            record.execution_time_ms = round(elapsed_ms, 2)
            record.success = False
            record.error_message = f"Tool execution hard timed out after {self.timeout_sec}s."
            record.results_count = 0
            return {
                "success": False,
                "data": [],
                "record": record,
                "error": record.error_message
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
