---
order: 8
title: "Enterprise RAG Assistant"
subtitle: "Multimodal document Q&A with citations"
category: "AI"
summary: "A document Q&A assistant with citation-backed retrieval that runs entirely inside one container — local embeddings, an embedded vector index, one external call for generation."
tech: ["LlamaIndex", "Fastembed (ONNX)", "OpenRouter", "FastAPI", "Python"]
---
### System Architecture
A Morphik-style document Q&A assistant built to deliver high-fidelity information retrieval backed by verifiable source citations. Retrieval and embedding run inside the API container itself; the only network call in the request path is the one that words the answer.

### Production Implementation
- **Ingestion, at build time:** A build step (`ingest.py`) reads the markdown corpus, chunks it, embeds it, and persists the resulting vector index into the image. Ingestion is not a service, a cron job, or a manual upload — it is a layer of the container.
- **Embeddings, locally:** [Fastembed](https://github.com/qdrant/fastembed) runs `BAAI/bge-small-en-v1.5` (384 dimensions) through ONNX Runtime on plain CPU. No GPU, no embedding endpoint. The model weights are pulled during the image build and the container starts with `HF_HUB_OFFLINE=1`, so a missing weight fails the build instead of failing the first query after a cold start.
- **Vector storage, embedded:** A LlamaIndex `SimpleVectorStore` persisted as JSON next to the code (~130 KB for the whole corpus) and read from that directory at query time. There is no cluster to connect to, no connection string, and nothing that can be reclaimed for inactivity.
- **Generation, one external call:** `llama-index-llms-openrouter` reaches `openai/gpt-4o-mini` through OpenRouter's OpenAI-schema endpoint, so the assistant has one credential and one bill for generation instead of a second vendor relationship.
- **Query path:** the incoming question is embedded by the same local model, the three closest corpus sections are pulled out of the on-disk index, and the model writes the answer from that context.
- **Serving:** The same FastAPI container on Cloud Run with the same `/api/v1/query` contract and the same application-level rate limit — the frontend widget did not change.

### Why the hosted dependencies were removed
The first version kept its vectors in a free-tier Qdrant Cloud cluster and called OpenAI directly for generation. The cluster was dismantled for disuse — nobody deleted it, it simply stopped existing — and every query began answering with a dropped connection. The service was healthy, the image was unchanged, the pipeline was green, and the assistant was down: a dependency that fails silently is worse than one that fails loudly. Retrieval can no longer be dismantled from a distance, because retrieval is now local code reading local files.

> **Live Demo Notice:** The interactive RAG widget embedded below runs against a fixed, pre-ingested, and curated demo corpus baked into the deployed image. It enforces strict application-level rate limits and compute budget caps to preserve operational safety.
