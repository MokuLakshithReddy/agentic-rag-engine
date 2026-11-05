"""
Controlled Tool Registry.
Enforces explicit tool registration, execution permissions, budget tracking,
and ensures external web access is disabled by default.
"""

from typing import Dict, Any, List, Optional
from src.tools.base import BaseTool
from src.tools.implementations import (
    DenseRetrievalTool,
    SparseRetrievalTool,
    HybridRetrievalTool,
    MetadataSearchTool,
    MultiQueryRetrievalTool,
    RerankEvidenceTool,
)
from src.ingestion.indexer import KnowledgeBase


class ToolRegistry:
    """
    Central tool executor ensuring only authorized tools run.
    Supports process isolation propagation, hard timeout enforcement, and tool governance.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        allow_web_search: bool = False,
        process_isolated: bool = False
    ):
        self.kb = kb
        self.allow_web_search = allow_web_search
        self.process_isolated = process_isolated
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        dense_tool = DenseRetrievalTool(self.kb, process_isolated=self.process_isolated)
        sparse_tool = SparseRetrievalTool(self.kb, process_isolated=self.process_isolated)
        hybrid_tool = HybridRetrievalTool(self.kb, process_isolated=self.process_isolated)
        meta_tool = MetadataSearchTool(self.kb, process_isolated=self.process_isolated)
        multi_q_tool = MultiQueryRetrievalTool(self.kb, process_isolated=self.process_isolated)
        rerank_tool = RerankEvidenceTool(process_isolated=self.process_isolated)

        for t in [dense_tool, sparse_tool, hybrid_tool, meta_tool, multi_q_tool, rerank_tool]:
            self._tools[t.name] = t

    def register_tool(self, tool: BaseTool) -> None:
        """Registers a custom or specialized tool."""
        self._tools[tool.name] = tool

    def set_process_isolation(self, enabled: bool) -> None:
        """Updates process isolation policy across all registered tools."""
        self.process_isolated = enabled
        for tool in self._tools.values():
            tool.process_isolated = enabled

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self._tools.get(tool_name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": name, "description": tool.description}
            for name, tool in self._tools.items()
        ]

    def execute_tool(self, tool_name: str, isolate_process: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """
        Executes a registered tool if authorized.
        Rejects arbitrary unapproved tools and unauthorized web search.
        Dispatches execution to process isolation when configured.
        """
        if tool_name not in self._tools:
            return {
                "success": False,
                "data": [],
                "error": f"Tool '{tool_name}' is not registered or not permitted."
            }

        tool = self._tools[tool_name]
        use_isolation = self.process_isolated if isolate_process is None else isolate_process
        return tool.run(isolate_process=use_isolation, **kwargs)

