"""Build the embedded vector index that ships inside the container image.

Reads the markdown corpus in `data/`, embeds it locally with a CPU-only ONNX
model (no embedding API call), and persists a LlamaIndex `SimpleVectorStore` to
the directory `app/engine.py` loads from. Runs as a build step in the Dockerfile,
so the image carries both the model weights and the finished index and neither is
fetched at runtime.

    python ingest.py            # from the rag-backend directory

Overridable paths (defaults match what the engine reads):

    RAG_INDEX_DIR        where the index is persisted   (default ./index_store)
    RAG_EMBED_CACHE_DIR  where model weights are cached (default ./model_cache)
    RAG_EMBED_MODEL      fastembed model name           (default BAAI/bge-small-en-v1.5)
"""
import os
import time
from pathlib import Path

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding

BACKEND_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BACKEND_ROOT / "data"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BACKEND_ROOT / "index_store"))
EMBED_CACHE_DIR = Path(os.getenv("RAG_EMBED_CACHE_DIR", BACKEND_ROOT / "model_cache"))
EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_THREADS = int(os.getenv("RAG_EMBED_THREADS", "2"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))


def read_corpus(data_dir: Path) -> list[Document]:
    """One Document per markdown file, keeping the file name as retrieval metadata."""
    documents = []
    for path in sorted(data_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                text=text,
                metadata={"source": path.name, "project": path.stem},
            )
        )
    return documents


def run_ingestion() -> None:
    print(f"Reading corpus from {DATA_DIR} ...")
    documents = read_corpus(DATA_DIR)
    if not documents:
        raise SystemExit(f"No markdown files found under {DATA_DIR}")
    print(f"Loaded {len(documents)} documents.")

    print(f"Loading local embedding model {EMBED_MODEL_NAME} (CPU/ONNX)...")
    started = time.perf_counter()
    embed_model = FastEmbedEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_dir=str(EMBED_CACHE_DIR),
        threads=EMBED_THREADS,
    )
    Settings.embed_model = embed_model
    print(f"Model ready in {time.perf_counter() - started:.1f}s")

    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    started = time.perf_counter()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(),
        show_progress=True,
    )
    print(f"Embedded and indexed in {time.perf_counter() - started:.1f}s")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    print(f"Persisted index to {INDEX_DIR}:")
    for path in sorted(INDEX_DIR.iterdir()):
        print(f"  {path.name:34s} {path.stat().st_size / 1024:8.1f} KiB")


if __name__ == "__main__":
    run_ingestion()
