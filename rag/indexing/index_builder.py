"""
RAG Index Builder
Author: Akash Hiremath (1RF23CS013)
Week: 3 — RAG Implementation

Purpose:
    Extracts text from medical PDFs, splits into chunks,
    generates embeddings and stores them in a FAISS index.
"""

import os
import json
import pickle
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

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
]

# ── Text Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list:
    """Extract text page by page from a PDF using PyMuPDF."""
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_num + 1
            })
    doc.close()
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Index Building ────────────────────────────────────────────────────────────

def build_index():
    """
    Main pipeline:
    1. Scan corpus for PDFs
    2. Extract text from each PDF
    3. Split into chunks
    4. Embed with all-MiniLM-L6-v2
    5. Store in FAISS index + metadata pickle
    """
    print("=== RAG Index Builder ===\n")

    # Initialize embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Model loaded: {EMBEDDING_MODEL}\n")

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    all_chunks = []
    all_metadata = []

    for category in CATEGORIES:
        category_path = os.path.join(CORPUS_DIR, category)
        if not os.path.exists(category_path):
            continue

        for filename in os.listdir(category_path):
            if not filename.endswith(".pdf"):
                continue

            pdf_path = os.path.join(category_path, filename)
            print(f"Processing: {filename} ({category})")

            pages = extract_text_from_pdf(pdf_path)
            print(f"  Extracted {len(pages)} pages")

            file_chunks = 0
            for page_data in pages:
                chunks = split_into_chunks(page_data["text"])
                for chunk in chunks:
                    if len(chunk.strip()) < 50:
                        continue

                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": filename,
                        "category": category,
                        "page": page_data["page"]
                    })
                    file_chunks += 1

            print(f"  Indexed {file_chunks} chunks\n")

    print(f"Total chunks: {len(all_chunks)}")
    print("Generating embeddings (this may take a few minutes)...")

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_STORE_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

    print(f"\n=== Indexing Complete ===")
    print(f"Total chunks indexed: {len(all_chunks)}")
    print(f"FAISS index saved to: {VECTOR_STORE_DIR}/index.faiss")
    print(f"Metadata saved to: {VECTOR_STORE_DIR}/metadata.pkl")


if __name__ == "__main__":
    build_index()