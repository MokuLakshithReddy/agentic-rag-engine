"""
Unit tests for text chunker and document loader.
"""

import pytest
from src.ingestion.chunker import RecursiveTextChunker
from src.ingestion.loader import DocumentLoader


def test_recursive_chunker_basic():
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Paragraph one is about distributed consensus. "
        "Paragraph two discusses Byzantine fault tolerance in asynchronous systems. "
        "Paragraph three covers sharding."
    )
    chunks = chunker.chunk_document("doc_test", sample_text)
    assert len(chunks) > 0
    for c in chunks:
        assert c.doc_id == "doc_test"
        assert c.chunk_id.startswith("doc_test#c")
        assert len(c.content) > 0


def test_chunker_empty_string():
    chunker = RecursiveTextChunker()
    chunks = chunker.chunk_document("empty_doc", "")
    assert chunks == []


def test_document_loader_unsupported_extension():
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        loader.load_file("malicious.exe")

