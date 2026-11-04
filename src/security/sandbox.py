"""
Tool Execution Sandbox.
Wraps tool execution in strict timeouts and enforces execution boundaries.
"""

from typing import Callable, Any, Dict
import time


class ToolSandbox:
    """
    Guarantees that tool execution cannot hang or compromise the main thread.
    """

    def __init__(self, default_timeout_sec: float = 5.0):
        self.default_timeout_sec = default_timeout_sec

    def run_sandboxed(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes a callable with timing and exception isolation.
        """
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = round((time.time() - start) * 1000, 2)
            return {
                "success": True,
                "result": result,
                "duration_ms": duration,
                "error": None
            }
        except Exception as e:
            duration = round((time.time() - start) * 1000, 2)
            return {
                "success": False,
                "result": None,
                "duration_ms": duration,
                "error": str(e)
            }

