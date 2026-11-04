"""
Curated Evaluation Benchmark Dataset.
Provides test cases across simple factual, comparative, multi-document,
and unanswerable (abstention) questions.
"""

from typing import List, Dict, Any


SAMPLE_KNOWLEDGE_CORPUS = [
    {
        "doc_id": "doc_distributed_consensus",
        "category": "distributed_systems",
        "text": (
            "In distributed consensus, the Raft protocol achieves consensus through leader election and log replication. "
            "Raft breaks consensus into three distinct subproblems: leader election, log replication, and safety. "
            "A cluster of 2F+1 nodes can tolerate F node failures. In contrast, Paxos allows any node to propose values, "
            "often leading to dual-leader contention unless Multi-Paxos with a stable leader is employed."
        )
    },
    {
        "doc_id": "doc_byzantine_fault_tolerance",
        "category": "distributed_systems",
        "text": (
            "Practical Byzantine Fault Tolerance (PBFT) provides Byzantine fault tolerance in asynchronous systems. "
            "PBFT requires 3F+1 total nodes to tolerate F Byzantine (arbitrary or malicious) faulty nodes. "
            "PBFT operates in three phases: pre-prepare, prepare, and commit. "
            "Its message complexity is O(N^2), which limits scalability to dozens of validator nodes."
        )
    },
    {
        "doc_id": "doc_solana_architecture",
        "category": "blockchain",
        "text": (
            "Solana utilizes Proof of History (PoH) as a cryptographic clock before consensus. "
            "PoH sequences transactions sequentially using SHA-256 verifiable delay functions. "
            "Solana's consensus engine is Tower BFT, an optimized variation of PBFT that leverages PoH to achieve "
            "sub-second block times (~400 milliseconds) and optimistic transaction finality without massive voting message overhead."
        )
    },
    {
        "doc_id": "doc_ethereum_pos",
        "category": "blockchain",
        "text": (
            "Ethereum uses a Proof of Stake mechanism called Gasper, combining Casper FFG (Friendly Finality Gadget) "
            "and LMD-GHOST (Latest Message Driven GHOST) fork choice rule. "
            "Ethereum validators must stake 32 ETH to propose and attest to blocks. "
            "Finality in Gasper requires two consecutive justified epochs, which typically takes 2 epochs (~12.8 minutes). "
            "Ethereum can coordinate hundreds of thousands of active validators through attestation aggregation."
        )
    },
    {
        "doc_id": "doc_database_indexing",
        "category": "databases",
        "text": (
            "B-Tree indexes maintain sorted keys in balanced multi-way tree nodes, providing O(log N) lookup, insertion, "
            "and range scan performance. In contrast, Log-Structured Merge (LSM) trees optimize for write-heavy workloads "
            "by buffering writes in an in-memory MemTable before appending them sequentially to disk in immutable SSTables."
        )
    }
]


BENCHMARK_QUERIES = [
    {
        "query_id": "q1_simple_raft",
        "query": "How does Raft break consensus into subproblems?",
        "category": "simple_factual",
        "expected_doc_ids": ["doc_distributed_consensus"],
        "is_answerable": True,
        "ground_truth": "Raft breaks consensus into three distinct subproblems: leader election, log replication, and safety."
    },
    {
        "query_id": "q2_comparative_eth_sol",
        "query": "Compare Ethereum and Solana consensus mechanisms regarding finality and validator architecture.",
        "category": "comparative",
        "expected_doc_ids": ["doc_ethereum_pos", "doc_solana_architecture"],
        "is_answerable": True,
        "ground_truth": "Ethereum uses Gasper (Casper FFG and LMD-GHOST) requiring 32 ETH and ~12.8 minutes for finality. Solana uses Proof of History with Tower BFT achieving ~400ms finality."
    },
    {
        "query_id": "q3_multi_doc_pbft_raft",
        "query": "What is the fault tolerance threshold difference between Raft and PBFT?",
        "category": "multi_document",
        "expected_doc_ids": ["doc_distributed_consensus", "doc_byzantine_fault_tolerance"],
        "is_answerable": True,
        "ground_truth": "Raft requires 2F+1 nodes to tolerate F crash failures, whereas PBFT requires 3F+1 nodes to tolerate F Byzantine failures."
    },
    {
        "query_id": "q4_simple_indexing",
        "query": "Why do LSM trees perform better than B-Trees on write-heavy workloads?",
        "category": "simple_factual",
        "expected_doc_ids": ["doc_database_indexing"],
        "is_answerable": True,
        "ground_truth": "LSM trees buffer writes in an in-memory MemTable before sequential append to immutable SSTables on disk."
    },
    {
        "query_id": "q5_unanswerable_quantum",
        "query": "What quantum error correction code is used in the Falcon processor?",
        "category": "out_of_scope",
        "expected_doc_ids": [],
        "is_answerable": False,
        "ground_truth": "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
    }
]

