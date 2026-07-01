"""
RAG Retriever — Improved
Author: Akash
Week: 5 — RAG Improvement & Knowledge Validation

Improvements over Week 4:
- Confidence scores for each retrieved chunk
- Duplicate chunk removal
- Better ranking using score normalization
- Category-based filtering option
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

VECTOR_STORE_DIR = "rag/vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6          # retrieve more, then re-rank to top 4
FINAL_TOP_K = 4    # final number returned after dedup + ranking
SIMILARITY_THRESHOLD = 0.8  # chunks more similar than this are duplicates


# ── Query Builder ─────────────────────────────────────────────────────────────

def build_query(risk_level: str, ecg_class: str = None,
                ef_value: float = None) -> str:
    """
    Build a clinically meaningful search query from fusion output.
    """
    query_parts = []

    risk_queries = {
        "Low": "primary prevention cardiovascular disease lifestyle recommendations",
        "Medium": "moderate cardiovascular risk management treatment guidelines",
        "High": "high cardiovascular risk treatment intervention guidelines"
    }
    query_parts.append(risk_queries.get(risk_level, risk_queries["Medium"]))

    if ecg_class:
        ecg_queries = {
            "NORM": "normal ECG cardiovascular risk assessment",
            "MI": "myocardial infarction ECG findings treatment",
            "STTC": "ST segment T wave changes cardiovascular management",
            "CD": "conduction disturbance bundle branch block management",
            "HYP": "cardiac hypertrophy hypertension ECG findings treatment"
        }
        if ecg_class in ecg_queries:
            query_parts.append(ecg_queries[ecg_class])

    if ef_value is not None:
        if ef_value < 40:
            query_parts.append(
                "heart failure reduced ejection fraction HFrEF management"
            )
        elif ef_value < 55:
            query_parts.append(
                "mildly reduced ejection fraction cardiac monitoring"
            )

    return " ".join(query_parts)


# ── Confidence Score Calculator ───────────────────────────────────────────────

def calculate_confidence(distance: float, max_distance: float = 10.0) -> float:
    """
    Convert FAISS L2 distance to a 0-1 confidence score.
    Lower distance = higher confidence.
    """
    confidence = max(0.0, 1.0 - (distance / max_distance))
    return round(confidence, 3)


# ── Duplicate Remover ─────────────────────────────────────────────────────────

def remove_duplicates(chunks: list, threshold: float = SIMILARITY_THRESHOLD) -> list:
    """
    Remove near-duplicate chunks based on text overlap ratio.

    Args:
        chunks: list of chunk dicts
        threshold: overlap ratio above which a chunk is considered duplicate

    Returns:
        Deduplicated list of chunks
    """
    unique_chunks = []
    seen_texts = []

    for chunk in chunks:
        text = chunk["text"].lower()
        is_duplicate = False

        for seen in seen_texts:
            # Calculate word overlap ratio
            words_a = set(text.split())
            words_b = set(seen.split())
            if len(words_a | words_b) == 0:
                continue
            overlap = len(words_a & words_b) / len(words_a | words_b)
            if overlap > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_chunks.append(chunk)
            seen_texts.append(text)

    return unique_chunks


# ── Retriever ─────────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Improved RAG Retriever with confidence scores,
    deduplication and better ranking.
    """

    def __init__(self):
        print("Initializing RAG Retriever...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
        metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                "FAISS index not found. Run index_builder.py first."
            )

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]

        print(f"Loaded {len(self.chunks)} chunks from vector store.")
        print("RAG Retriever ready.\n")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """
        Retrieve top-k chunks with confidence scores,
        then deduplicate and return final top results.
        """
        embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(embedding, top_k)

        # Calculate max distance for normalization
        max_dist = float(distances[0].max()) if distances[0].max() > 0 else 10.0

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            distance = float(distances[0][i])
            confidence = calculate_confidence(distance, max_dist)

            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "category": self.metadata[idx]["category"],
                "page": self.metadata[idx]["page"],
                "distance": round(distance, 4),
                "confidence": confidence
            })

        # Sort by confidence descending
        results.sort(key=lambda x: x["confidence"], reverse=True)

        # Remove duplicates
        results = remove_duplicates(results)

        # Return final top K
        return results[:FINAL_TOP_K]

    def retrieve_for_prediction(self, risk_level: str,
                                 ecg_class: str = None,
                                 ef_value: float = None) -> dict:
        """
        Full pipeline: build query → retrieve → return with metadata.
        """
        query = build_query(risk_level, ecg_class, ef_value)
        print(f"Query: {query}")

        chunks = self.retrieve(query)
        print(f"Retrieved {len(chunks)} chunks (after dedup)\n")

        return {
            "query": query,
            "chunks": chunks
        }


# ── Entry Point (Test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Improved RAG Retriever Test ===\n")

    retriever = RAGRetriever()

    print("Test: High risk + MI + EF 35%")
    result = retriever.retrieve_for_prediction(
        risk_level="High",
        ecg_class="MI",
        ef_value=35.0
    )
    for i, chunk in enumerate(result["chunks"]):
        print(f"  Chunk {i+1}: [{chunk['source']} p.{chunk['page']}] "
              f"confidence={chunk['confidence']}")
        print(f"    {chunk['text'][:80]}...")