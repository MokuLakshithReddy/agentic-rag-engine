"""
Base Tool definition with genuinely bounded execution timeouts,
capacity limits, cooperative cancellation tokens, and process isolation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import time
import threading
import multiprocessing
import concurrent.futures
from pydantic import BaseModel, Field
from src.state.state import ToolCallRecord, EvidenceChunk


class BaseToolInput(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


def _process_target_wrapper(queue: multiprocessing.Queue, func: Callable, args: tuple, kwargs: dict) -> None:
    """Standalone target executed inside a process-isolated worker."""
    try:
        res = func(*args, **kwargs)
        queue.put({"success": True, "data": res, "error": None})
    except Exception as e:
        queue.put({"success": False, "data": None, "error": str(e)})


def _standalone_tool_execute(tool: Any, kwargs: dict) -> List[EvidenceChunk]:
    """
    Top-level module function for executing tool logic inside an isolated process
    without module or closure resolution failures across Windows/Linux.
    """
    return tool._execute(**kwargs)


def execute_with_process_isolation(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout_sec: float = 5.0
) -> Dict[str, Any]:
    """
    Executes a function inside an isolated operating system process.
    If the operation exceeds timeout_sec, the process is forcefully terminated
    (SIGTERM / TerminateProcess) and killed, releasing all OS resources.
    """
    if kwargs is None:
        kwargs = {}

    ctx = multiprocessing.get_context()
    queue = ctx.Queue()
    proc = ctx.Process(target=_process_target_wrapper, args=(queue, func, args, kwargs))
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=0.1)
        return {
            "success": False,
            "data": None,
            "error": f"Process hard-killed after exceeding {timeout_sec}s timeout."
        }

    if not queue.empty():
        return queue.get_nowait()
    return {"success": False, "data": None, "error": "Process completed without output."}


class BaseTool(ABC):
    """
    Abstract base tool class.
    Executes operations under hard, non-blocking timeout guarantees,
    worker capacity limits, and cooperative cancellation tokens.
    """

    MAX_CONCURRENT_WORKERS = 8
    MAX_PENDING_QUEUE = 16
    MAX_CONCURRENT_PROCESSES = 4
    MAX_PENDING_PROCESS_QUEUE = 8

    _shared_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    _active_tasks: int = 0
    _active_processes: int = 0
    _lock = threading.Lock()

    @classmethod
    def _get_executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        if cls._shared_executor is None:
            cls._shared_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=cls.MAX_CONCURRENT_WORKERS,
                thread_name_prefix="rag_tool_worker"
            )
        return cls._shared_executor

    def __init__(
        self,
        name: str,
        description: str,
        timeout_sec: float = 5.0,
        process_isolated: bool = False
    ):
        self.name = name
        self.description = description
        self.timeout_sec = timeout_sec
        self.process_isolated = process_isolated

    @abstractmethod
    def _execute(self, **kwargs) -> List[EvidenceChunk]:
        """Tool specific execution logic."""
        pass

    def run_isolated(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the tool inside an isolated OS process with hard process termination on timeout
        and strictly bounded process concurrency limits.
        """
        start_time = time.time()
        record = ToolCallRecord(
            tool_name=self.name,
            query_input=str(kwargs.get("query", kwargs.get("subqueries", "")))[:200]
        )

        # Enforce strict process capacity limit
        with self._lock:
            if self._active_processes >= (self.MAX_CONCURRENT_PROCESSES + self.MAX_PENDING_PROCESS_QUEUE):
                record.execution_time_ms = 0.0
                record.success = False
                record.error_message = (
                    f"Process capacity exceeded: worker process pool saturated "
                    f"({self._active_processes} active/queued). Execution rejected."
                )
                return {
                    "success": False,
                    "data": [],
                    "record": record,
                    "error": record.error_message
                }
            BaseTool._active_processes += 1

        try:
            res = execute_with_process_isolation(
                _standalone_tool_execute,
                args=(self, kwargs),
                timeout_sec=self.timeout_sec
            )
            elapsed_ms = (time.time() - start_time) * 1000.0
            record.execution_time_ms = round(elapsed_ms, 2)
            if res.get("success"):
                data = res.get("data", [])
                record.success = True
                record.results_count = len(data)
                return {"success": True, "data": data, "record": record}
            else:
                record.success = False
                record.error_message = res.get("error", f"Tool hard timed out or failed in isolated process after {self.timeout_sec}s.")
                record.results_count = 0
                return {"success": False, "data": [], "record": record, "error": record.error_message}
        finally:
            with self._lock:
                BaseTool._active_processes = max(0, BaseTool._active_processes - 1)

    def run(self, isolate_process: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """
        Executes the tool with bounded concurrency, cancellation tracking, and timeout guarantees.
        When isolate_process is True (or self.process_isolated is True), delegates to run_isolated().
        """
        use_isolation = self.process_isolated if isolate_process is None else isolate_process
        if use_isolation:
            return self.run_isolated(**kwargs)

        start_time = time.time()
        record = ToolCallRecord(
            tool_name=self.name,
            query_input=str(kwargs.get("query", kwargs.get("subqueries", "")))[:200]
        )

        # 1. Enforce strict capacity limit (active + pending queue)
        with self._lock:
            if self._active_tasks >= (self.MAX_CONCURRENT_WORKERS + self.MAX_PENDING_QUEUE):
                record.execution_time_ms = 0.0
                record.success = False
                record.error_message = (
                    f"Tool capacity exceeded: worker pool saturated "
                    f"({self._active_tasks} tasks running/queued). Execution rejected."
                )
                return {
                    "success": False,
                    "data": [],
                    "record": record,
                    "error": record.error_message
                }
            BaseTool._active_tasks += 1

        # 2. Cooperative cancellation token passed to worker
        cancel_token = threading.Event()
        exec_kwargs = dict(kwargs)
        exec_kwargs["cancel_token"] = cancel_token

        executor = self._get_executor()
        future = executor.submit(self._execute, **exec_kwargs)

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
            # Signal worker to abort immediately and cancel queued future
            cancel_token.set()
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
            cancel_token.set()
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
        finally:
            with self._lock:
                BaseTool._active_tasks = max(0, BaseTool._active_tasks - 1)
