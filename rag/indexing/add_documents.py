"""Append new PDFs from rag/corpus/<category>/ to the existing FAISS index."""
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.indexing.index_builder import extract_text_from_pdf, split_into_chunks
from rag.model_config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION

CORPUS_DIR = Path("rag/corpus")
VECTOR_STORE_DIR = Path("rag/vector_store")
CHUNK_SIZE = 200      # words; fits MiniLM's 256-token limit
CHUNK_OVERLAP = 40


def main():
    index_path = VECTOR_STORE_DIR / "index.faiss"
    meta_path = VECTOR_STORE_DIR / "metadata.pkl"

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    chunks, metadata = data["chunks"], data["metadata"]
    assert len(chunks) == len(metadata) == index.ntotal, "Index/metadata mismatch"

    already_indexed = {m["source"] for m in metadata}
    new_chunks, new_meta = [], []

    for category_dir in sorted(p for p in CORPUS_DIR.iterdir() if p.is_dir()):
        for pdf in sorted(category_dir.glob("*.pdf")):
            if pdf.name in already_indexed:
                print(f"Skipping (already indexed): {pdf.name}")
                continue
            print(f"Processing: {pdf.name} ({category_dir.name})")
            count = 0
            for page in extract_text_from_pdf(str(pdf)):
                for chunk in split_into_chunks(page["text"], CHUNK_SIZE, CHUNK_OVERLAP):
                    if len(chunk.strip()) < 50:
                        continue
                    new_chunks.append(chunk)
                    new_meta.append({
                        "source": pdf.name,
                        "category": category_dir.name,
                        "page": page["page"],
                    })
                    count += 1
            print(f"  {count} chunks")

    if not new_chunks:
        print("No new PDFs found.")
        return

    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION)
    embeddings = model.encode(new_chunks, show_progress_bar=True).astype("float32")
    assert embeddings.shape[1] == index.d, "Embedding dimension mismatch"

    index.add(embeddings)
    chunks.extend(new_chunks)
    metadata.extend(new_meta)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)

    print(f"\nAdded {len(new_chunks)} chunks. Total now: {index.ntotal}")


if __name__ == "__main__":
    main()