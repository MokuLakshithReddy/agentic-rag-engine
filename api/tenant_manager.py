"""
Tenant Knowledge Base and Workflow Isolation Manager with Path Traversal Defense.
Provides isolated storage directories, knowledge base instances, tool registries,
and agent workflows for each distinct user organization, tenant, or guest sandbox.
Strictly validates tenant IDs against injection and filesystem breakout attacks.
"""

from typing import Dict, Any, Optional
import re
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
    Guarantees that tenant IDs cannot escape their designated sandbox storage directory.
    """

    TENANT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{3,64}$")

    def __init__(self, base_data_dir: str = "./data", default_process_isolation: bool = True):
        self.base_data_dir = Path(base_data_dir).resolve()
        self.default_process_isolation = default_process_isolation
        self._lock = threading.Lock()
        self._kbs: Dict[str, KnowledgeBase] = {}
        self._workflows: Dict[str, AgenticRAGWorkflow] = {}

    def _validate_and_resolve_storage(self, tenant_id: str) -> Path:
        """
        Validates tenant ID format and verifies the resolved filesystem path remains
        strictly contained within the authorized tenant root directory.
        """
        tid = tenant_id.strip() if tenant_id else "default"
        if not self.TENANT_ID_REGEX.match(tid):
            raise ValueError(
                f"Invalid tenant_id format: '{tid}'. "
                f"Tenant IDs must contain only alphanumeric characters, hyphens, and underscores (3-64 chars)."
            )

        base_resolved = self.base_data_dir.resolve()
        if tid == "default":
            storage_dir = (base_resolved / "indices").resolve()
            if not storage_dir.is_relative_to(base_resolved):
                raise ValueError(f"Path traversal detected: '{storage_dir}' escapes base directory '{base_resolved}'.")
        else:
            tenants_root = (base_resolved / "tenants").resolve()
            tenants_root.mkdir(parents=True, exist_ok=True)
            storage_dir = (tenants_root / tid / "indices").resolve()
            if not storage_dir.is_relative_to(tenants_root):
                raise ValueError(
                    f"Path traversal attack detected: '{storage_dir}' is outside tenant root '{tenants_root}'."
                )

        return storage_dir

    def get_kb(self, tenant_id: str = "default") -> KnowledgeBase:
        tid = tenant_id.strip() if tenant_id else "default"
        storage_dir = self._validate_and_resolve_storage(tid)

        with self._lock:
            if tid in self._kbs:
                return self._kbs[tid]

            kb = KnowledgeBase(storage_dir=str(storage_dir))

            # Initialize tenant with standard knowledge corpus if freshly created
            if not kb.raw_documents:
                for item in SAMPLE_KNOWLEDGE_CORPUS:
                    kb.index_document(
                        doc_id=item["doc_id"],
                        text=item["text"],
                        metadata={"category": item["category"]}
                    )

            self._kbs[tid] = kb
            return kb

    def get_workflow(
        self,
        tenant_id: str = "default",
        process_isolated: Optional[bool] = None
    ) -> AgenticRAGWorkflow:
        tid = tenant_id.strip() if tenant_id else "default"
        # Validate tenant ID safety
        self._validate_and_resolve_storage(tid)

        use_isolation = self.default_process_isolation if process_isolated is None else process_isolated

        with self._lock:
            if tid in self._workflows:
                return self._workflows[tid]

        # Outside lock to prevent lock inversion with KnowledgeBase internal locks
        kb = self.get_kb(tid)
        registry = ToolRegistry(kb, process_isolated=use_isolation)
        workflow = AgenticRAGWorkflow(registry, process_isolated=use_isolation)

        with self._lock:
            self._workflows[tid] = workflow
            return workflow
