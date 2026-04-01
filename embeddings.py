"""
Embedding service using sentence-transformers (all-MiniLM-L6-v2).
Provides a singleton model to encode texts into dense vectors.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# Lazy-loaded singleton
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("[Embeddings] Model loaded.")
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized embeddings.

    Args:
        texts: List of strings to encode.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), L2-normalized.
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def encode_single(text: str) -> np.ndarray:
    """Encode a single text string and return its embedding vector."""
    return encode([text])[0]
