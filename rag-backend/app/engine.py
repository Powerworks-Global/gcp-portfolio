# app/engine.py
"""Retrieval and generation for the portfolio assistant, with no hosted data store.

Three things used to be external here: the vector database, the embedding model and
the generation model. Retrieval and embedding are now in-process — the index is
persisted into the image at build time by `ingest.py` and read back from that same
directory — and only the generated answer leaves the container, via OpenRouter's
OpenAI-schema chat endpoint.
"""
import os
from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openrouter import OpenRouter

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Weights and index are build-time artifacts (see ingest.py / Dockerfile). Pinning
# HF_HUB_OFFLINE=1 in the image means a missing weight fails loudly instead of
# reaching for the network on a cold start.
EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE_DIR = os.getenv("RAG_EMBED_CACHE_DIR", str(BACKEND_ROOT / "model_cache"))
EMBED_THREADS = int(os.getenv("RAG_EMBED_THREADS", "2"))
INDEX_DIR = os.getenv("RAG_INDEX_DIR", str(BACKEND_ROOT / "index_store"))

GENERATION_MODEL = os.getenv("RAG_GENERATION_MODEL", "openai/gpt-4o-mini")
GENERATION_CONTEXT_WINDOW = int(os.getenv("RAG_GENERATION_CONTEXT_WINDOW", "128000"))
# Stated here rather than inherited so the address the assistant talks to is visible in
# this file; overriding it is how the query path gets exercised against a local stand-in.
GENERATION_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")


class RAGEngine:
    def __init__(self):
        self._query_engine = None

    def _get_engine(self):
        if self._query_engine is None:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

            if not openrouter_api_key:
                raise ValueError(
                    "Missing required environment variable: OPENROUTER_API_KEY"
                )

            if not (Path(INDEX_DIR) / "index_store.json").is_file():
                raise FileNotFoundError(
                    f"No vector index at {INDEX_DIR}. It is built during the image "
                    "build (python ingest.py); rebuild the image, or run ingest.py "
                    "locally before starting the API."
                )

            Settings.embed_model = FastEmbedEmbedding(
                model_name=EMBED_MODEL_NAME,
                cache_dir=EMBED_CACHE_DIR,
                threads=EMBED_THREADS,
            )

            llm = OpenRouter(
                model=GENERATION_MODEL,
                temperature=0.1,
                max_tokens=512,
                context_window=GENERATION_CONTEXT_WINDOW,
                api_base=GENERATION_API_BASE,
                api_key=openrouter_api_key,
            )
            Settings.llm = llm

            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=INDEX_DIR)
            )
            self._query_engine = index.as_query_engine(similarity_top_k=3, llm=llm)
        return self._query_engine

    def query(self, prompt: str) -> str:
        engine = self._get_engine()
        response = engine.query(prompt)
        return str(response)


# LAZY INSTANCE: Singleton created only when needed, not on module import
_engine_instance = None


def get_rag_engine() -> RAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance
