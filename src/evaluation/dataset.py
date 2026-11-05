"""
Curated Evaluation Benchmark Dataset.
Provides an expanded, statistically meaningful corpus of 25 domain documents
and 24 rigorous test queries across simple factual, comparative, multi-document,
and diverse unanswerable (abstention) failure modes.
"""

from typing import List, Dict, Any


SAMPLE_KNOWLEDGE_CORPUS = [
    # --- Distributed Systems & Consensus ---
    {
        "doc_id": "doc_distributed_consensus",
        "category": "distributed_systems",
        "text": (
            "In distributed consensus, the Raft protocol achieves consensus through leader election and log replication. "
            "Raft breaks consensus into three distinct subproblems: leader election, log replication, and safety. "
            "A cluster of 2F+1 nodes can tolerate F crash failures. In contrast, Paxos allows any node to propose values, "
            "often leading to dual-leader contention unless Multi-Paxos with a stable leader is employed."
        )
    },
    {
        "doc_id": "doc_raft_leader_election",
        "category": "distributed_systems",
        "text": (
            "Raft leader election uses randomized election timeouts between 150ms and 300ms to prevent split votes. "
            "A follower transitions to candidate state, increments its current term, votes for itself, and issues "
            "RequestVote RPCs to all peers. If it receives votes from a majority of nodes, it becomes the term leader "
            "and immediately sends periodic heartbeats to assert leadership."
        )
    },
    {
        "doc_id": "doc_raft_log_replication",
        "category": "distributed_systems",
        "text": (
            "During Raft log replication, the leader appends client commands as new entries and broadcasts AppendEntries RPCs. "
            "An entry is considered committed once it is replicated on a majority of cluster nodes. "
            "The Raft Log Matching Property guarantees that if two logs contain an entry with the same index and term, "
            "then the logs are identical in all entries up through the given index."
        )
    },
    {
        "doc_id": "doc_byzantine_fault_tolerance",
        "category": "distributed_systems",
        "text": (
            "Practical Byzantine Fault Tolerance (PBFT) provides Byzantine fault tolerance in asynchronous systems. "
            "PBFT requires 3F+1 total nodes to tolerate F Byzantine (arbitrary or malicious) faulty nodes. "
            "PBFT operates in three distinct phases: pre-prepare, prepare, and commit. "
            "Its message complexity is O(N^2), which limits scalability to dozens of validator nodes."
        )
    },
    {
        "doc_id": "doc_tendermint_consensus",
        "category": "distributed_systems",
        "text": (
            "Tendermint BFT is a deterministic leader-based consensus algorithm designed for state-machine replication. "
            "Validators take turns proposing blocks and voting on them in rounds consisting of propose, prevote, and precommit steps. "
            "A block requires greater than two-thirds (+2/3) of voting power in both prevote and precommit stages to commit, "
            "guaranteeing instant finality without chain forks under less than 1/3 Byzantine voting power."
        )
    },
    {
        "doc_id": "doc_avalanche_consensus",
        "category": "distributed_systems",
        "text": (
            "Avalanche consensus uses repeated sub-sampled voting over a metastable network to achieve fast finality. "
            "Instead of polling all validators like PBFT, a node randomly queries a small constant sample size k of peers. "
            "If a threshold alpha of peers agree, the node updates its preferred choice. Over successive rounds, "
            "the network experiences positive feedback, converging rapidly to consensus with O(N log N) message efficiency."
        )
    },

    # --- Blockchain Architectures ---
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
        "doc_id": "doc_bitcoin_nakamoto",
        "category": "blockchain",
        "text": (
            "Bitcoin uses Nakamoto consensus combining Proof of Work (PoW) with the heaviest-chain rule (highest cumulative difficulty). "
            "Miners compute SHA-256 double-hashes against a target difficulty adjusted every 2016 blocks (~2 weeks). "
            "Bitcoin produces blocks approximately every 10 minutes and achieves probabilistic finality, typically requiring "
            "6 block confirmations (~60 minutes) to render double-spend reorganization computationally infeasible."
        )
    },

    # --- Database Storage & Internals ---
    {
        "doc_id": "doc_database_indexing",
        "category": "databases",
        "text": (
            "B-Tree indexes maintain sorted keys in balanced multi-way tree nodes, providing O(log N) lookup, insertion, "
            "and range scan performance. In contrast, Log-Structured Merge (LSM) trees optimize for write-heavy workloads "
            "by buffering writes in an in-memory MemTable before appending them sequentially to disk in immutable SSTables."
        )
    },
    {
        "doc_id": "doc_columnar_storage",
        "category": "databases",
        "text": (
            "Columnar storage formats like Apache Parquet and ORC store table data by column rather than row. "
            "This layout enables high compression ratios using techniques such as Run-Length Encoding (RLE) and dictionary encoding. "
            "Analytical query engines benefit from SIMD vectorized execution and projection pushdown, reading only columns relevant to the query."
        )
    },
    {
        "doc_id": "doc_write_ahead_logging",
        "category": "databases",
        "text": (
            "Write-Ahead Logging (WAL) ensures durability and atomicity in relational databases through the ARIES recovery algorithm. "
            "Before changes are applied to in-memory buffer pool data pages, log records describing the modification must be flushed to non-volatile disk. "
            "During crash recovery, ARIES performs three passes: Analysis (reconstructing dirty pages), Redo (repeating history), and Undo (rolling back uncommitted transactions)."
        )
    },
    {
        "doc_id": "doc_bloom_filters",
        "category": "databases",
        "text": (
            "A Bloom filter is a space-efficient probabilistic data structure used to test set membership. "
            "It can yield false positives (reporting an element is in the set when it is not) but never false negatives. "
            "Storage engines like RocksDB place Bloom filters on SSTables to avoid expensive disk lookups for non-existent keys."
        )
    },

    # --- Concurrency & Transactions ---
    {
        "doc_id": "doc_acid_transactions",
        "category": "transactions",
        "text": (
            "ACID properties govern database transactional integrity: Atomicity (all-or-nothing), Consistency (state invariant preservation), "
            "Isolation (concurrent transaction separation), and Durability (committed survival across crashes). "
            "SQL standard isolation levels from weakest to strongest are Read Uncommitted, Read Committed, Repeatable Read, and Serializable. "
            "Serializable isolation prevents anomalies including dirty reads, non-repeatable reads, phantom reads, and write skew."
        )
    },
    {
        "doc_id": "doc_multi_version_concurrency",
        "category": "transactions",
        "text": (
            "Multi-Version Concurrency Control (MVCC) eliminates read locks by maintaining multiple physical versions of each record. "
            "Readers access a consistent historical snapshot identified by transaction timestamps, allowing readers to never block writers, "
            "and writers to never block readers. PostgreSQL implements MVCC using xmin/xmax transaction IDs and vacuum processes to purge dead tuples."
        )
    },
    {
        "doc_id": "doc_distributed_transactions_2pc",
        "category": "transactions",
        "text": (
            "Two-Phase Commit (2PC) coordinates atomic commitment across multiple distributed database shards. "
            "In Phase 1 (Prepare), the coordinator asks all participants whether they can commit. If all answer yes, "
            "the coordinator proceeds to Phase 2 (Commit); otherwise, it issues an Abort. 2PC is a blocking protocol: "
            "if the coordinator crashes during Phase 2, participants remain indefinitely in doubt holding locks."
        )
    },
    {
        "doc_id": "doc_sharding_partitioning",
        "category": "databases",
        "text": (
            "Database sharding horizontally partitions data across separate machine instances. "
            "Range-based partitioning divides rows by ordered key boundaries, which enables efficient range scans but risks hotspots. "
            "Hash-based partitioning applies a hash function across shard keys to distribute writes evenly, but requires broadcast queries for multi-shard range scans."
        )
    },
    {
        "doc_id": "doc_cap_theorem_pacelc",
        "category": "distributed_systems",
        "text": (
            "Brewer's CAP theorem states that a distributed data store can simultaneously provide at most two of three guarantees: "
            "Consistency (linearizability), Availability (every request receives a non-error response), and Partition tolerance (network splits). "
            "The PACELC theorem extends CAP: If there is a Partition (P), how does the system trade Availability (A) and Consistency (C)? "
            "Else (E), how does the system trade Latency (L) and Consistency (C)?"
        )
    },

    # --- Cryptography & Security ---
    {
        "doc_id": "doc_zero_knowledge_proofs",
        "category": "cryptography",
        "text": (
            "Zero-Knowledge Proofs (ZKPs) allow a prover to convince a verifier that a statement is true without revealing any secret information. "
            "zk-SNARKs (Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge) require an initial trusted setup ceremony and have short proof sizes (~200 bytes). "
            "zk-STARKs (Scalable Transparent Arguments of Knowledge) eliminate trusted setup using collision-resistant hash functions, but produce larger proof sizes (~50-100 KB)."
        )
    },
    {
        "doc_id": "doc_tls_cryptography",
        "category": "cryptography",
        "text": (
            "TLS 1.3 reduces the cryptographic handshake latency to a single round-trip (1-RTT) by performing Ephemeral Diffie-Hellman key exchange "
            "alongside the initial ClientHello message. TLS 1.3 deprecates legacy ciphers (RSA key exchange, CBC mode ciphers) and mandates "
            "Authenticated Encryption with Associated Data (AEAD) algorithms such as AES-GCM and ChaCha20-Poly1305 to ensure forward secrecy."
        )
    },
    {
        "doc_id": "doc_threshold_signatures",
        "category": "cryptography",
        "text": (
            "Threshold signature schemes like BLS (Boneh-Lynn-Shacham) and Shamir-based threshold ECDSA distribute signing authority among N parties. "
            "Any subset of T-of-N participants can collaborate to create a single valid signature, while fewer than T parties learn nothing about the secret key. "
            "BLS signatures feature non-interactive aggregation, allowing thousands of validator signatures to be compressed into a single 48-byte signature."
        )
    },

    # --- Search & Vector Systems ---
    {
        "doc_id": "doc_vector_search_hnsw",
        "category": "search_systems",
        "text": (
            "Hierarchical Navigable Small World (HNSW) graphs perform Approximate Nearest Neighbor (ANN) search in high-dimensional vector spaces. "
            "HNSW organizes vectors into a multi-layer graph where upper layers have long-range skip edges for rapid coarse routing, "
            "and the bottom layer contains dense local links for fine-grained convergence. It achieves logarithmic O(log N) search complexity."
        )
    },
    {
        "doc_id": "doc_vector_quantization_ivfpq",
        "category": "search_systems",
        "text": (
            "Inverted File with Product Quantization (IVF-PQ) scales vector search to billion-scale datasets by compressing vectors into short byte codes. "
            "The IVF component clusters vectors into Voronoi cells to prune non-candidate vectors during query time. "
            "Product Quantization decomposes D-dimensional vectors into M sub-vectors and quantizes each sub-vector using small codebooks, "
            "reducing memory footprint by up to 95%."
        )
    },
    {
        "doc_id": "doc_distributed_caching",
        "category": "distributed_systems",
        "text": (
            "Distributed caching architectures like Redis Cluster and Memcached reduce primary database load. "
            "Cache-aside pattern requires the application to query the cache first; upon a cache miss, the app fetches from DB and populates cache. "
            "Consistent hashing with virtual nodes maps cache keys uniformly across server nodes, ensuring that adding or removing a cache node "
            "only migrates K/N keys rather than invalidating the entire cluster."
        )
    },
    {
        "doc_id": "doc_gossip_protocol",
        "category": "distributed_systems",
        "text": (
            "Gossip protocols (epidemic algorithms) achieve decentralized peer-to-peer state synchronization and failure detection without a central leader. "
            "Each node periodically selects a random peer and exchanges membership lists or cluster metadata. "
            "Information spreads exponentially across N nodes in O(log N) time rounds, providing high fault tolerance against packet loss and node crashes."
        )
    }
]


BENCHMARK_QUERIES = [
    # --- Category 1: Simple Factual (6 queries) ---
    {
        "query_id": "q_fact_1_raft_subproblems",
        "query": "How does Raft break consensus into subproblems?",
        "category": "simple_factual",
        "expected_doc_ids": ["doc_distributed_consensus"],
        "is_answerable": True,
        "ground_truth": "Raft breaks consensus into three distinct subproblems: leader election, log replication, and safety."
    },
    {
        "query_id": "q_fact_2_raft_timeouts",
        "query": "What election timeout range does Raft use to prevent split votes?",
        "category": "simple_factual",
        "expected_doc_ids": ["doc_raft_leader_election"],
        "is_answerable": True,
        "ground_truth": "Raft uses randomized election timeouts between 150ms and 300ms."
    },
    {
        "query_id": "q_fact_3_lsm_trees",
        "query": "Why do LSM trees perform better than B-Trees on write-heavy workloads?",
        "category": "simple_factual",
        "expected_doc_ids": ["doc_database_indexing"],
        "is_answerable": True,
        "ground_truth": "LSM trees buffer writes in an in-memory MemTable before sequentially appending to immutable SSTables on disk."
    },
    {
        "query_id": "q_fact_4_aries_passes",
        "query": "What are the three passes performed by the ARIES recovery algorithm in WAL?",
        "category": "simple_factual",
        "expected_doc_ids": ["doc_write_ahead_logging"],
        "is_answerable": True,
        "ground_truth": "ARIES performs three passes: Analysis, Redo, and Undo."
    },
    {
        "query_id": "q_fact_5_hnsw_routing",
        "query": "How does the HNSW graph achieve logarithmic search complexity for vector embeddings?",
        "category": "simple_factual",
        "expected_doc_ids": ["doc_vector_search_hnsw"],
        "is_answerable": True,
        "ground_truth": "HNSW organizes vectors into a multi-layer graph where upper layers have long-range skip edges for coarse routing and the bottom layer has dense local links."
    },
    {
        "query_id": "q_fact_6_tls_rtt",
        "query": "How does TLS 1.3 reduce handshake latency to a single round-trip?",
        "category": "simple_factual",
        "expected_doc_ids": ["doc_tls_cryptography"],
        "is_answerable": True,
        "ground_truth": "TLS 1.3 performs Ephemeral Diffie-Hellman key exchange alongside the initial ClientHello message to achieve 1-RTT handshake latency."
    },

    # --- Category 2: Comparative Analysis (6 queries) ---
    {
        "query_id": "q_comp_1_eth_vs_sol",
        "query": "Compare Ethereum and Solana consensus mechanisms regarding finality and validator architecture.",
        "category": "comparative",
        "expected_doc_ids": ["doc_ethereum_pos", "doc_solana_architecture"],
        "is_answerable": True,
        "ground_truth": "Ethereum uses Gasper with 32 ETH stakes and ~12.8 minute epoch finality, while Solana uses Proof of History with Tower BFT achieving ~400ms finality."
    },
    {
        "query_id": "q_comp_2_raft_vs_paxos",
        "query": "Compare Raft and Paxos regarding proposal handling and leader contention.",
        "category": "comparative",
        "expected_doc_ids": ["doc_distributed_consensus"],
        "is_answerable": True,
        "ground_truth": "Raft uses a single stable elected leader for log replication, whereas basic Paxos allows any node to propose values leading to dual-leader contention unless Multi-Paxos is used."
    },
    {
        "query_id": "q_comp_3_snarks_vs_starks",
        "query": "What are the trade-offs between zk-SNARKs and zk-STARKs regarding trusted setups and proof sizes?",
        "category": "comparative",
        "expected_doc_ids": ["doc_zero_knowledge_proofs"],
        "is_answerable": True,
        "ground_truth": "zk-SNARKs require a trusted setup ceremony and have small proof sizes (~200 bytes), while zk-STARKs require no trusted setup but produce larger proof sizes (~50-100 KB)."
    },
    {
        "query_id": "q_comp_4_btree_vs_lsm",
        "query": "Compare B-Tree indexes and LSM trees regarding read and write optimization trade-offs.",
        "category": "comparative",
        "expected_doc_ids": ["doc_database_indexing"],
        "is_answerable": True,
        "ground_truth": "B-Trees maintain sorted balanced trees optimized for reads and range scans (O(log N)), while LSM trees buffer in memory for append-only sequential writes."
    },
    {
        "query_id": "q_comp_5_range_vs_hash_sharding",
        "query": "Compare range-based partitioning and hash-based partitioning in distributed database sharding.",
        "category": "comparative",
        "expected_doc_ids": ["doc_sharding_partitioning"],
        "is_answerable": True,
        "ground_truth": "Range-based partitioning enables efficient range scans but risks hotspots, while hash partitioning distributes writes evenly but requires cross-shard broadcast queries."
    },
    {
        "query_id": "q_comp_6_tendermint_vs_pbft",
        "query": "Compare the quorum requirements and validator scalability of Tendermint BFT and classical PBFT.",
        "category": "comparative",
        "expected_doc_ids": ["doc_tendermint_consensus", "doc_byzantine_fault_tolerance"],
        "is_answerable": True,
        "ground_truth": "PBFT requires 3F+1 nodes with O(N^2) messages limiting to dozens of nodes, while Tendermint uses +2/3 voting power across rounds to achieve instant finality."
    },

    # --- Category 3: Multi-Document / Multi-Hop Reasoning (6 queries) ---
    {
        "query_id": "q_multi_1_pbft_vs_raft_faults",
        "query": "What is the fault tolerance threshold difference between Raft and PBFT in terms of node count?",
        "category": "multi_document",
        "expected_doc_ids": ["doc_distributed_consensus", "doc_byzantine_fault_tolerance"],
        "is_answerable": True,
        "ground_truth": "Raft requires 2F+1 nodes to tolerate F crash failures, whereas PBFT requires 3F+1 nodes to tolerate F Byzantine failures."
    },
    {
        "query_id": "q_multi_2_mvcc_wal_durability",
        "query": "How do databases combine MVCC snapshots and Write-Ahead Logging to achieve non-blocking reads with durability?",
        "category": "multi_document",
        "expected_doc_ids": ["doc_multi_version_concurrency", "doc_write_ahead_logging"],
        "is_answerable": True,
        "ground_truth": "MVCC provides readers with consistent timestamped versions without read locks, while WAL flushes modifications before disk commit to guarantee ARIES crash recovery."
    },
    {
        "query_id": "q_multi_3_blockchain_finality_times",
        "query": "How do the block production and finality times differ across Bitcoin, Ethereum, and Solana?",
        "category": "multi_document",
        "expected_doc_ids": ["doc_bitcoin_nakamoto", "doc_ethereum_pos", "doc_solana_architecture"],
        "is_answerable": True,
        "ground_truth": "Bitcoin produces blocks in ~10m with ~60m finality; Ethereum uses Gasper with ~12.8m finality; Solana uses PoH with Tower BFT for ~400ms block finality."
    },
    {
        "query_id": "q_multi_4_cap_vs_2pc",
        "query": "Why does Two-Phase Commit prioritize Consistency over Availability during network partitions under the CAP theorem?",
        "category": "multi_document",
        "expected_doc_ids": ["doc_distributed_transactions_2pc", "doc_cap_theorem_pacelc"],
        "is_answerable": True,
        "ground_truth": "In 2PC, if the coordinator or network partitions during Phase 2, participants block holding locks to preserve consistency, sacrificing availability."
    },
    {
        "query_id": "q_multi_5_bloom_filter_lsm",
        "query": "How do Bloom filters prevent unnecessary disk access in LSM-tree SSTables?",
        "category": "multi_document",
        "expected_doc_ids": ["doc_database_indexing", "doc_bloom_filters"],
        "is_answerable": True,
        "ground_truth": "Because LSM trees append data to immutable SSTables, Bloom filters test key non-membership to skip querying SSTables that do not contain the target key."
    },
    {
        "query_id": "q_multi_6_vector_indexing_tradeoffs",
        "query": "How do HNSW and IVF-PQ approach the trade-off between memory footprint and search speed in vector databases?",
        "category": "multi_document",
        "expected_doc_ids": ["doc_vector_search_hnsw", "doc_vector_quantization_ivfpq"],
        "is_answerable": True,
        "ground_truth": "HNSW builds multi-layer graph structures for fast O(log N) search at the expense of higher memory, while IVF-PQ uses Voronoi clustering and sub-vector quantization to cut memory by up to 95%."
    },

    # --- Category 4: Unanswerable & Abstention Verification (6 queries across diverse failure modes) ---
    {
        "query_id": "q_abst_1_unindexed_domain_quantum",
        "query": "What quantum error correction code is used in the Falcon processor?",
        "category": "out_of_scope",
        "expected_doc_ids": [],
        "is_answerable": False,
        "ground_truth": "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
    },
    {
        "query_id": "q_abst_2_unindexed_domain_biology",
        "query": "What lipid nanoparticle formulation is used to encapsulate mRNA in modern vaccines?",
        "category": "out_of_scope",
        "expected_doc_ids": [],
        "is_answerable": False,
        "ground_truth": "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
    },
    {
        "query_id": "q_abst_3_secret_credentials",
        "query": "What is the production private validator signing key for the primary Ethereum validator cluster?",
        "category": "out_of_scope",
        "expected_doc_ids": [],
        "is_answerable": False,
        "ground_truth": "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
    },
    {
        "query_id": "q_abst_4_false_premise_contradiction",
        "query": "Explain how the Raft consensus protocol tolerates malicious Byzantine faults and arbitrary validator collusions.",
        "category": "out_of_scope",
        "expected_doc_ids": [],
        "is_answerable": False,
        "ground_truth": "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
    },
    {
        "query_id": "q_abst_5_future_temporal_mismatch",
        "query": "What are the exact finalized protocol parameters and validator slot counts for Ethereum in the year 2050?",
        "category": "out_of_scope",
        "expected_doc_ids": [],
        "is_answerable": False,
        "ground_truth": "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
    },
    {
        "query_id": "q_abst_6_nonexistent_configuration",
        "query": "How do you configure the --enable-infinite-throughput flag in PostgreSQL B-Tree indexes?",
        "category": "out_of_scope",
        "expected_doc_ids": [],
        "is_answerable": False,
        "ground_truth": "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
    }
]
