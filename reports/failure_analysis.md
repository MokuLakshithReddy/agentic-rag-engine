# Failure Analysis & Edge Case Evaluation

## 1. Evaluated Edge Cases

- **Out-of-Corpus Query (q5_unanswerable_quantum)**: Correctly routed through the Evidence Evaluator as NO_EVIDENCE and triggered safe abstention without hallucinating citations.
- **Multi-Entity Comparative Query (q2_comparative_eth_sol)**: Decomposed into subqueries targeting both architectures independently, avoiding single-entity bias.
- **Write-Heavy Storage Query (q4_simple_indexing)**: Grounded in LSM-tree MemTable and SSTable evidence with valid citation attached.
- **Adversarial Content Injection**: Prompt injection directives inside retrieved evidence are stripped and delimited in untrusted context tags.
