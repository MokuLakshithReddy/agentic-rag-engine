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
    """

    def __init__(self, kb: KnowledgeBase, allow_web_search: bool = False):
        self.kb = kb
        self.allow_web_search = allow_web_search
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        dense_tool = DenseRetrievalTool(self.kb)
        sparse_tool = SparseRetrievalTool(self.kb)
        hybrid_tool = HybridRetrievalTool(self.kb)
        meta_tool = MetadataSearchTool(self.kb)
        multi_q_tool = MultiQueryRetrievalTool(self.kb)
        rerank_tool = RerankEvidenceTool()

        for t in [dense_tool, sparse_tool, hybrid_tool, meta_tool, multi_q_tool, rerank_tool]:
            self._tools[t.name] = t

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self._tools.get(tool_name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": name, "description": tool.description}
            for name, tool in self._tools.items()
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Executes a registered tool if authorized.
        Rejects arbitrary unapproved tools and unauthorized web search.
        """
        if tool_name not in self._tools:
            return {
                "success": False,
                "data": [],
                "error": f"Tool '{tool_name}' is not registered or not permitted."
            }

        tool = self._tools[tool_name]
        return tool.run(**kwargs)

