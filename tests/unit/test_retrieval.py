"""
Unit tests for Dense, Sparse, Hybrid, and Metadata retrieval engines.
"""

import pytest
from src.state.state import EvidenceChunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.metadata import MetadataRetriever


@pytest.fixture
def sample_chunks():
    return [
        EvidenceChunk(chunk_id="doc1#c0", doc_id="doc1", content="Raft consensus election safety log replication.", metadata={"category": "consensus"}),
        EvidenceChunk(chunk_id="doc2#c0", doc_id="doc2", content="Solana uses Proof of History and Tower BFT.", metadata={"category": "blockchain"}),
        EvidenceChunk(chunk_id="doc3#c0", doc_id="doc3", content="Postgres relational database indexes with B-Tree.", metadata={"category": "database"}),
    ]


def test_dense_retrieval(sample_chunks):
    dense = DenseRetriever(top_k=2)
    dense.build_index(sample_chunks)
    results = dense.search("Tell me about Solana and Proof of History")
    assert len(results) > 0
    assert results[0].doc_id == "doc2"
    assert results[0].score > 0.0


def test_sparse_retrieval(sample_chunks):
    sparse = SparseRetriever(top_k=2)
    sparse.build_index(sample_chunks)
    results = sparse.search("Postgres B-Tree")
    assert len(results) > 0
    assert results[0].doc_id == "doc3"


def test_hybrid_rrf_retrieval(sample_chunks):
    dense = DenseRetriever(top_k=3)
    dense.build_index(sample_chunks)
    sparse = SparseRetriever(top_k=3)
    sparse.build_index(sample_chunks)

    hybrid = HybridRetriever(dense, sparse, top_k=2)
    results = hybrid.search("Raft consensus algorithm")
    assert len(results) > 0
    assert results[0].doc_id == "doc1"
    assert results[0].strategy == "hybrid_retrieval"


def test_metadata_search(sample_chunks):
    meta = MetadataRetriever(top_k=2)
    meta.set_chunks(sample_chunks)
    results = meta.search(category="blockchain")
    assert len(results) == 1
    assert results[0].doc_id == "doc2"

