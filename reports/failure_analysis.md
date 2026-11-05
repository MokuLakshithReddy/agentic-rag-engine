# Failure Analysis & Edge Case Evaluation

## 1. Evaluated Edge Case Categories (24 Benchmark Queries)

- **Unindexed Knowledge Domains**: Queries regarding quantum processor error correction and mRNA lipid nanoparticles correctly routed through the Evidence Evaluator as NO_EVIDENCE, triggering safe abstention without hallucinated citations.
- **Confidential Operational Data**: Queries soliciting private production validator keys safely abstained due to zero supporting evidence in the knowledge corpus.
- **False Premise Questions**: Inquiries asking to explain how crash-fault protocols (Raft) tolerate Byzantine malicious attacks detected as contradictory / unsupported, triggering abstention.
- **Multi-Entity Comparative Queries**: Complex queries comparing Ethereum vs Solana, Raft vs Paxos, and B-Tree vs LSM-Tree decomposed into targeted subqueries, avoiding single-entity retrieval bias.
- **Adversarial Content Injection**: Direct and indirect prompt injection attempts inside query text and evidence passages stripped and scrubbed by the security layer.
