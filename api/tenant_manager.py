"""
Tenant Knowledge Base and Workflow Isolation Manager.
Provides isolated storage directories, knowledge base instances, tool registries,
and agent workflows for each distinct user organization, tenant, or guest sandbox.
"""

from typing import Dict, Any, Optional
import threading
from pathlib import Path
from src.ingestion.indexer import KnowledgeBase
from src.tools.registry import ToolRegistry
from src.agents.workflow import AgenticRAGWorkflow
from src.evaluation.dataset import SAMPLE_KNOWLEDGE_CORPUS


class TenantKnowledgeBaseManager:
    """
    Manages tenant-isolated KnowledgeBase instances and associated research workflows.
    Ensures that documents ingested, queried, or deleted by one tenant are strictly partitioned
    from all other tenants and guest sandboxes.
    """

    def __init__(self, base_data_dir: str = "./data"):
        self.base_data_dir = Path(base_data_dir).resolve()
        self._lock = threading.Lock()
        self._kbs: Dict[str, KnowledgeBase] = {}
        self._workflows: Dict[str, AgenticRAGWorkflow] = {}

    def get_kb(self, tenant_id: str = "default") -> KnowledgeBase:
        tid = tenant_id.strip() or "default"
        with self._lock:
            if tid in self._kbs:
                return self._kbs[tid]

            if tid == "default":
                storage_dir = self.base_data_dir / "indices"
            else:
                storage_dir = self.base_data_dir / "tenants" / tid / "indices"

            kb = KnowledgeBase(storage_dir=str(storage_dir))

            # Initialize tenant with sample knowledge corpus if freshly created
            if not kb.raw_documents:
                for item in SAMPLE_KNOWLEDGE_CORPUS:
                    kb.index_document(
                        doc_id=item["doc_id"],
                        text=item["text"],
                        metadata={"category": item["category"]}
                    )

            self._kbs[tid] = kb
            return kb

    def get_workflow(self, tenant_id: str = "default") -> AgenticRAGWorkflow:
        tid = tenant_id.strip() or "default"
        with self._lock:
            if tid in self._workflows:
                return self._workflows[tid]

        # Outside lock to prevent lock inversion with KnowledgeBase locks
        kb = self.get_kb(tid)
        registry = ToolRegistry(kb)
        workflow = AgenticRAGWorkflow(registry)

        with self._lock:
            self._workflows[tid] = workflow
            return workflow
