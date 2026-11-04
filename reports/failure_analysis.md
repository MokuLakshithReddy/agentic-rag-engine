# Failure Analysis & Edge Case Evaluation

## 1. Analyzed Edge Cases

- **Ambiguous Terminology**: Queries with underspecified context are classified as AMBIGUOUS and fall back to broader hybrid searches.
- **Out-of-Corpus Questions**: Quantum computing test query correctly triggered the abstention safeguard without unverified hallucinations.
- **Indirect Prompt Injection in Retrieved Docs**: Malicious directives embedded inside retrieved documents were neutralized by strict delimiters and content-instruction separation.
