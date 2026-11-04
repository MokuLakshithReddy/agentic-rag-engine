"""
Document chunker module.
Splits documents into coherent passages while maintaining chunk identifiers,
document provenance, page offsets, and deterministic hashes.
"""

from typing import List, Dict, Any, Optional
import re
import hashlib
from src.state.state import EvidenceChunk


class RecursiveTextChunker:
    """
    Recursively splits text using paragraph, sentence, and whitespace separators
    to maintain context coherence and bounded chunk token lengths.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " "]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        sep = separators[0]
        splits = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in splits:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > self.chunk_size and len(separators) > 1:
                    sub_chunks = self._split_text(part, separators[1:])
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current and current.strip():
            chunks.append(current.strip())

        return chunks

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> List[EvidenceChunk]:
        """
        Chunks raw document content into validated EvidenceChunk instances.
        """
        metadata = metadata or {}
        cleaned_text = re.sub(r"\r\n|\r", "\n", text).strip()
        if not cleaned_text:
            return []

        raw_splits = self._split_text(cleaned_text, self.separators)
        evidence_chunks: List[EvidenceChunk] = []

        for idx, chunk_str in enumerate(raw_splits):
            if not chunk_str.strip():
                continue
            chunk_hash = hashlib.sha256(f"{doc_id}_{idx}_{chunk_str[:50]}".encode("utf-8")).hexdigest()[:8]
            chunk_id = f"{doc_id}#c{idx}_{chunk_hash}"

            chunk_meta = dict(metadata)
            chunk_meta["chunk_index"] = idx
            chunk_meta["char_length"] = len(chunk_str)

            evidence_chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk_str,
                    page_number=page_number,
                    metadata=chunk_meta
                )
            )

        return evidence_chunks

