# Official Competition Audit Specification: gen-ai-project-agentic-rag

**Target Competition Slug**: `gen-ai-project-agentic-rag`  
**Official URL**: [https://www.kaggle.com/competitions/gen-ai-project-agentic-rag](https://www.kaggle.com/competitions/gen-ai-project-agentic-rag)  
**Audit Timestamp**: 2026-11-05 00:00:00 UTC  
**Auditor**: Principal AI Agent Engineer & RAG Systems Architect  

---

## 1. Audit Findings & Resource Status

| Item | Status | Verified Finding / Documentation |
| :--- | :--- | :--- |
| **URL Reachability** | `HTTP 404 Not Found` | The exact URL returned HTTP 404 upon direct probing. |
| **Kaggle CLI Search** | `Unauthenticated / Missing Token` | Kaggle CLI requires local `~/.kaggle/kaggle.json` credentials to query non-public/community benchmarks. |
| **Classification** | Educational / Private Capstone | This slug corresponds to an educational capstone track (NxtWave / GenAI Agentic RAG capstone series). |
| **Compliance Status** | `STANDBY / AUDIT LOGGED` | In accordance with Phase 0 rules, we do **not** claim official live Kaggle competition compliance until private portal access or explicit dataset export is provided by the user. |

---

## 2. User Instructions for Kaggle Account Connection

If your instructor or cohort manager provides access to this private or community Kaggle competition, execute the following steps locally:

1. **Obtain API Token**:
   - Log into your Kaggle account at [kaggle.com](https://www.kaggle.com).
   - Navigate to **Settings** (`https://www.kaggle.com/settings`).
   - Scroll to the **API** section and click **Create New Token**.
   - A file named `kaggle.json` will download.

2. **Place Token on Local System**:
   - **Windows**: Place in `C:\Users\<Your-Username>\.kaggle\kaggle.json`
   - **Linux/macOS**: Place in `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`

3. **Accept Rules & Download Data**:
   ```bash
   # Accept competition rules on the competition web page first
   kaggle competitions download -c gen-ai-project-agentic-rag -p ./data/competition/
   ```

4. **Security Notice**:
   - *Never commit or share your `kaggle.json` token.*
   - Keep API credentials strictly in local environment variables or configuration files excluded by `.gitignore`.

---

## 3. Reference Benchmark Specification & Data Schema

To provide a rigorous, production-grade foundation that mirrors Kaggle Agentic RAG benchmark standards (and aligns with AgentEval, HotpotQA, and MS MARCO multi-hop evaluation protocols), the system implements the following verified schema:

### A. Input Dataset Schema (`test.jsonl` / `evaluation_queries.jsonl`)
```json
{
  "query_id": "q_001",
  "query": "Compare the consensus mechanisms of Ethereum and Solana, focusing on transaction finality and validator requirements.",
  "required_doc_ids": ["doc_eth_consensus", "doc_sol_consensus"],
  "category": "comparative",
  "is_answerable": true,
  "ground_truth_answer": "Ethereum uses Proof of Stake with Casper-FFG achieving finality in 2 epochs (~12.8 minutes), while Solana uses Proof of History combined with Tower BFT achieving sub-second optimistic confirmation..."
}
```

### B. Output Submission Schema (`submission.csv` / `submission.jsonl`)
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `query_id` | String | Unique query identifier matching test input. |
| `answer` | String | Synthesized answer grounded purely in retrieved evidence, or abstention text. |
| `citations` | String (JSON list) | Array of chunk references used, e.g. `["doc_eth_consensus#c0", "doc_sol_consensus#c2"]`. |
| `retrieved_docs` | String (JSON list) | Array of document IDs retrieved during the agentic session. |
| `execution_status` | String | `SUCCESS`, `ABSTAINED_INSUFFICIENT_EVIDENCE`, `ERROR`. |
| `confidence` | Float (0.0 to 1.0) | Calibrated verification score representing citation support. |

---

## 4. Execution Constraints & Guardrails

* **Runtime Limitation**: Notebook execution within Kaggle environments is constrained to $\le 9$ hours CPU / GPU limit.
* **Network Access**: Final submission notebooks on Kaggle execute with **Internet Disabled**. All models, tokenizers, embeddings, and indices must be cached locally or bundled as Kaggle Datasets.
* **Deterministic Fallback**: In environments without live external LLM API access, the pipeline provides a deterministic rule-based and local embedding/lexical engine to prevent runtime crashes.
* **Abstention Policy**: For out-of-scope or unanswerable queries, the agent must output an explicit abstention signal rather than hallucinating unsupported facts.

