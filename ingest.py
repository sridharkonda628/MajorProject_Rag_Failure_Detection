"""
Document ingestion script.
Reads .txt files from data/documents/, embeds them,
and saves the FAISS index to disk.
"""

import os
import sys
from vector_store import VectorStore
from config import DATA_DIR, FAISS_INDEX_DIR


def ingest_documents(data_dir: str | None = None) -> None:
    """
    Read all .txt files from the data directory,
    split them into chunks, embed, and save to FAISS index.
    """
    source_dir = data_dir or DATA_DIR

    if not os.path.exists(source_dir):
        print(f"[Ingest] Error: Data directory not found: {source_dir}")
        sys.exit(1)

    # Collect all .txt files
    txt_files = sorted([
        f for f in os.listdir(source_dir)
        if f.endswith(".txt")
    ])

    if not txt_files:
        print(f"[Ingest] No .txt files found in {source_dir}")
        sys.exit(1)

    print(f"[Ingest] Found {len(txt_files)} document(s) in {source_dir}")

    # Read and chunk documents
    all_chunks: list[str] = []
    for filename in txt_files:
        filepath = os.path.join(source_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        # Split into paragraphs (chunks)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        all_chunks.extend(paragraphs)
        print(f"  → {filename}: {len(paragraphs)} chunk(s)")

    print(f"[Ingest] Total chunks: {len(all_chunks)}")

    # Create vector store, embed, and save
    store = VectorStore()
    store.add_documents(all_chunks)
    store.save(FAISS_INDEX_DIR)

    print(f"[Ingest] ✅ Index saved to {FAISS_INDEX_DIR}")
    print(f"[Ingest] Total vectors: {store.index.ntotal}")


if __name__ == "__main__":
    ingest_documents()
