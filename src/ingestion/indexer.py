"""
Knowledge Base Indexer with Thread Synchronization and Local Persistence.
Orchestrates document chunking, indexing across Dense, Sparse, Hybrid,
and Metadata retrievers with RLock thread safety and disk persistence.
"""

from typing import List, Dict, Any, Optional
import os
import json
import threading
from pathlib import Path

from src.state.state import EvidenceChunk
from src.ingestion.chunker import RecursiveTextChunker
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.metadata import MetadataRetriever


class KnowledgeBase:
    """
    Thread-safe unified knowledge repository housing all evidence chunks.
    Supports in-memory mode or optional disk persistence when storage_dir is supplied.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        default_top_k: int = 5,
        storage_dir: Optional[str] = None
    ):
        self._lock = threading.RLock()
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.persist_file = self.storage_dir / "documents.jsonl"
        else:
            self.persist_file = None

        self.chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.dense = DenseRetriever(top_k=default_top_k)
        self.sparse = SparseRetriever(top_k=default_top_k)
        self.hybrid = HybridRetriever(self.dense, self.sparse, top_k=default_top_k)
        self.metadata = MetadataRetriever(top_k=default_top_k)

        # Document registry
        self.raw_documents: Dict[str, Dict[str, Any]] = {}
        self.all_chunks: List[EvidenceChunk] = []

        # Load persisted documents if configured
        if self.persist_file:
            self._load_from_disk()

    def __getstate__(self) -> Dict[str, Any]:
        """Custom serialization omitting threading locks for multiprocessing pickling."""
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom deserialization recreating re-entrant thread lock."""
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def _persist_to_disk(self) -> None:
        """Writes current raw documents to JSONL persistence file if enabled."""
        if not self.persist_file:
            return
        with open(self.persist_file, "w", encoding="utf-8") as f:
            for doc in self.raw_documents.values():
                f.write(json.dumps(doc) + "\n")

    def _load_from_disk(self) -> None:
        """Reads documents from JSONL file if exists."""
        if not self.persist_file or not self.persist_file.exists():
            return
        with open(self.persist_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    doc_id = doc["doc_id"]
                    text = doc["text"]
                    metadata = doc.get("metadata", {})
                    self.raw_documents[doc_id] = doc
                    chunks = self.chunker.chunk_document(doc_id=doc_id, text=text, metadata=metadata)
                    self.all_chunks.extend(chunks)
        if self.all_chunks:
            self._rebuild_indices()

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> List[EvidenceChunk]:
        """
        Thread-safely indexes a new document or updates an existing document.
        """
        with self._lock:
            metadata = metadata or {}

            # Remove existing chunks for this doc_id if updating
            if doc_id in self.raw_documents:
                self.remove_document(doc_id)

            doc_entry = {
                "doc_id": doc_id,
                "text": text,
                "metadata": metadata
            }
            self.raw_documents[doc_id] = doc_entry

            new_chunks = self.chunker.chunk_document(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                page_number=page_number
            )

            self.all_chunks.extend(new_chunks)
            self._rebuild_indices()
            self._persist_to_disk()
            return new_chunks

    def remove_document(self, doc_id: str) -> bool:
        """
        Thread-safely removes a document and its associated chunks from all indices.
        """
        with self._lock:
            if doc_id not in self.raw_documents:
                return False

            del self.raw_documents[doc_id]
            self.all_chunks = [c for c in self.all_chunks if c.doc_id != doc_id]
            self._rebuild_indices()
            self._persist_to_disk()
            return True

    def get_document_summary(self) -> List[Dict[str, Any]]:
        """
        Returns summary list of indexed documents.
        """
        with self._lock:
            summary = []
            for doc_id, doc_info in self.raw_documents.items():
                chunk_count = sum(1 for c in self.all_chunks if c.doc_id == doc_id)
                summary.append({
                    "doc_id": doc_id,
                    "chunk_count": chunk_count,
                    "metadata": doc_info["metadata"],
                    "preview": doc_info["text"][:150] + "..." if len(doc_info["text"]) > 150 else doc_info["text"]
                })
            return summary

    def _rebuild_indices(self) -> None:
        """
        Refreshes underlying vector, sparse, and metadata indices.
        """
        self.dense.build_index(self.all_chunks)
        self.sparse.build_index(self.all_chunks)
        self.metadata.set_chunks(self.all_chunks)
