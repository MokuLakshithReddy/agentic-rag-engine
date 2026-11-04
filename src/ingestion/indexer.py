"""
Knowledge Base Indexer.
Orchestrates document chunking, indexing across Dense, Sparse, Hybrid,
and Metadata retrievers, and provides thread-safe document mutations.
"""

from typing import List, Dict, Any, Optional
from src.state.state import EvidenceChunk
from src.ingestion.chunker import RecursiveTextChunker
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.metadata import MetadataRetriever


class KnowledgeBase:
    """
    Unified knowledge index repository housing all evidence chunks
    and active retrieval indices.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        default_top_k: int = 5
    ):
        self.chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.dense = DenseRetriever(top_k=default_top_k)
        self.sparse = SparseRetriever(top_k=default_top_k)
        self.hybrid = HybridRetriever(self.dense, self.sparse, top_k=default_top_k)
        self.metadata = MetadataRetriever(top_k=default_top_k)
        
        # Document registry
        self.raw_documents: Dict[str, Dict[str, Any]] = {}
        self.all_chunks: List[EvidenceChunk] = []

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> List[EvidenceChunk]:
        """
        Indexes a new document or updates an existing document by doc_id.
        """
        metadata = metadata or {}
        
        # Remove existing chunks for this doc_id if updating
        if doc_id in self.raw_documents:
            self.remove_document(doc_id)

        self.raw_documents[doc_id] = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata
        }

        new_chunks = self.chunker.chunk_document(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            page_number=page_number
        )

        self.all_chunks.extend(new_chunks)
        self._rebuild_indices()
        return new_chunks

    def remove_document(self, doc_id: str) -> bool:
        """
        Removes a document and its associated chunks from all indices.
        """
        if doc_id not in self.raw_documents:
            return False

        del self.raw_documents[doc_id]
        self.all_chunks = [c for c in self.all_chunks if c.doc_id != doc_id]
        self._rebuild_indices()
        return True

    def get_document_summary(self) -> List[Dict[str, Any]]:
        """
        Returns list of indexed documents with chunk counts and metadata.
        """
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

