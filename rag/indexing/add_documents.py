"""
Incremental Document Indexer
Author: Akash
Module 5 — Medical Knowledge Base Expansion

Purpose:
    Adds NEW documents to the existing FAISS index without rebuilding
    from scratch. Safe to run repeatedly — skips any source filename
    already present in the index, so it never duplicates or loses data.
"""

import os
import pickle
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_DIR = "rag/corpus"
VECTOR_STORE_DIR = "rag/vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

CATEGORIES = [
    "risk_factors",
    "lifestyle_recommendations",
    "echo_findings",
    "ecg_findings",
    "symptoms",
    "food",
    "hospital_references",
]


def extract_text_from_pdf(pdf_path: str) -> list:
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({"text": text.strip(), "page": page_num + 1})
    doc.close()
    return pages


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def add_new_documents():
    print("=== Incremental Document Indexer ===\n")

    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

    print("Loading existing index...")
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        store = pickle.load(f)

    existing_chunks = store["chunks"]
    existing_metadata = store["metadata"]
    existing_sources = set(m["source"] for m in existing_metadata)

    print(f"Existing index: {len(existing_chunks)} chunks from "
          f"{len(existing_sources)} documents\n")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    new_chunks = []
    new_metadata = []

    for category in CATEGORIES:
        category_path = os.path.join(CORPUS_DIR, category)
        if not os.path.exists(category_path):
            continue

        for filename in os.listdir(category_path):
            if not filename.endswith(".pdf"):
                continue

            if filename in existing_sources:
                print(f"Skipping (already indexed): {filename}")
                continue

            pdf_path = os.path.join(category_path, filename)
            print(f"Processing NEW document: {filename} ({category})")

            pages = extract_text_from_pdf(pdf_path)
            file_chunks = 0
            for page_data in pages:
                chunks = split_into_chunks(page_data["text"])
                for chunk in chunks:
                    if len(chunk.strip()) < 50:
                        continue
                    new_chunks.append(chunk)
                    new_metadata.append({
                        "source": filename,
                        "category": category,
                        "page": page_data["page"]
                    })
                    file_chunks += 1

            print(f"  Indexed {file_chunks} new chunks\n")

    if not new_chunks:
        print("No new documents found. Nothing to add.")
        return

    print(f"Total new chunks: {len(new_chunks)}")
    print("Generating embeddings for new chunks...")

    new_embeddings = model.encode(new_chunks, show_progress_bar=True)
    new_embeddings = np.array(new_embeddings).astype("float32")

    index.add(new_embeddings)

    all_chunks = existing_chunks + new_chunks
    all_metadata = existing_metadata + new_metadata

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

    print(f"\n=== Incremental Indexing Complete ===")
    print(f"Previous total: {len(existing_chunks)} chunks")
    print(f"Added: {len(new_chunks)} new chunks")
    print(f"New total: {len(all_chunks)} chunks")


if __name__ == "__main__":
    add_new_documents()