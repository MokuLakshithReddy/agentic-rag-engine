# Known Limitations and Production Considerations

Engineering transparency is a core tenet of this system. The following limitations are documented to guide future extensions and enterprise deployments.

---

## 1. Algorithmic and Retrieval Limitations

* **Cross-Lingual Retrieval**: The default embeddings and tokenization pipelines are optimized for English. Highly inflected or non-Latin alphabets require multilingual embedding models (e.g., `paraphrase-multilingual-mpnet-base-v2` or `text-embedding-3-large`).
* **Table & Image Modality**: Current chunking strategies treat tabular data as linearized markdown or plain text. High-density financial spreadsheets or architectural diagrams require specialized multimodal extractors (e.g., OCR / Table Transformer).
* **Cross-Encoder Computational Cost**: While cross-encoders provide high reranking precision, they incur significant CPU/GPU latency on large candidate pools ($K > 50$). For latency-sensitive SLAs ($< 500\text{ms}$), bi-encoder dot-product or lightweight ColBERT scoring is recommended.

---

## 2. Infrastructure & Competition Runtime Realities

* **Kaggle Kernel Offline Mode**: Kaggle competition scoring runs without external internet access. While the portfolio API can connect to OpenAI, Anthropic, or Hugging Face Inference Endpoints, the offline Kaggle notebook is configured to use local deterministic models or packaged weights.
* **Vector Index Persistence**: The local engine uses an in-memory/file-persisted vector and BM25 index. For multi-tenant systems serving millions of documents, migration to a distributed vector engine (e.g., Qdrant, Milvus, or pgvector) is standard.
* **Cold Start Latency**: First-time initialization of transformer embeddings requires model weight loading into memory. In warm production containers, this overhead is amortized across requests.

