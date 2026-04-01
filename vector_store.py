"""
FAISS-based vector store for document retrieval.
Supports adding documents, searching by query, and persistence.
"""

import os
import json
import numpy as np
import faiss

import embeddings as emb
from config import FAISS_INDEX_DIR


class VectorStore:
    """FAISS IndexFlatIP vector store with document text storage."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[str] = []
        self.dimension: int = 0

    def add_documents(self, texts: list[str]) -> None:
        """
        Embed and add documents to the FAISS index.

        Args:
            texts: List of document strings to add.
        """
        if not texts:
            return

        vectors = emb.encode(texts)
        dim = vectors.shape[1]

        if self.index is None:
            self.dimension = dim
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(vectors)
        self.documents.extend(texts)
        print(f"[VectorStore] Added {len(texts)} documents (total: {len(self.documents)})")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the vector store for the most relevant documents.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: 'text', 'score', 'index'.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = emb.encode_single(query).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "score": float(score),
                    "index": int(idx),
                })
        return results

    def save(self, directory: str | None = None) -> None:
        """Save the FAISS index and document list to disk."""
        save_dir = directory or FAISS_INDEX_DIR
        os.makedirs(save_dir, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))

        with open(os.path.join(save_dir, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        print(f"[VectorStore] Saved index and {len(self.documents)} documents to {save_dir}")

    def load(self, directory: str | None = None) -> None:
        """Load the FAISS index and document list from disk."""
        load_dir = directory or FAISS_INDEX_DIR

        index_path = os.path.join(load_dir, "index.faiss")
        docs_path = os.path.join(load_dir, "documents.json")

        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(
                f"FAISS index not found in {load_dir}. Run 'python ingest.py' first."
            )

        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d

        with open(docs_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        print(f"[VectorStore] Loaded index with {self.index.ntotal} vectors and {len(self.documents)} documents")


# ── Module-level convenience ─────────────────────────────────────────
_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Get or create the singleton VectorStore, loading from disk if available."""
    global _store
    if _store is None:
        _store = VectorStore()
        try:
            _store.load()
        except FileNotFoundError:
            print("[VectorStore] No existing index found. Starting with empty store.")
    return _store
