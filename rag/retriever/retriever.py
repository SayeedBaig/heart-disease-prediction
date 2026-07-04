"""
RAG Retriever — Validated
Author: Akash
Week: 6 — RAG Validation & Knowledge Expansion

Improvements over Week 5:
- Edge case handling for missing/insufficient matches
- Fallback query when retrieval returns poor results
- Minimum confidence threshold filtering
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = "rag/vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6
FINAL_TOP_K = 4
SIMILARITY_THRESHOLD = 0.8
MIN_CONFIDENCE = 0.01


def build_query(risk_level: str, ecg_class: str = None,
                ef_value: float = None) -> str:
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


def build_fallback_query(risk_level: str) -> str:
    """Fallback query when primary retrieval returns poor results."""
    return f"cardiovascular disease {risk_level.lower()} risk guidelines treatment"


def calculate_confidence(distance: float, max_distance: float = 10.0) -> float:
    confidence = max(0.0, 1.0 - (distance / max_distance))
    return round(confidence, 3)


def remove_duplicates(chunks: list,
                      threshold: float = SIMILARITY_THRESHOLD) -> list:
    unique_chunks = []
    seen_texts = []

    for chunk in chunks:
        text = chunk["text"].lower()
        is_duplicate = False

        for seen in seen_texts:
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


class RAGRetriever:

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
        embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(embedding, top_k)

        max_dist = float(distances[0].max()) if distances[0].max() > 0 else 10.0

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            distance = float(distances[0][i])
            confidence = calculate_confidence(distance, max_dist)

            if confidence < MIN_CONFIDENCE:
                continue

            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "category": self.metadata[idx]["category"],
                "page": self.metadata[idx]["page"],
                "distance": round(distance, 4),
                "confidence": confidence
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        results = remove_duplicates(results)
        return results[:FINAL_TOP_K]

    def retrieve_for_prediction(self, risk_level: str,
                                 ecg_class: str = None,
                                 ef_value: float = None) -> dict:
        query = build_query(risk_level, ecg_class, ef_value)
        print(f"Query: {query}")

        chunks = self.retrieve(query)

        # Edge case: if retrieval returns too few results, use fallback query
        if len(chunks) < 2:
            print("  WARNING: Insufficient results. Using fallback query...")
            fallback_query = build_fallback_query(risk_level)
            print(f"  Fallback query: {fallback_query}")
            chunks = self.retrieve(fallback_query)

        # Edge case: if still no results
        if not chunks:
            print("  WARNING: No relevant chunks found.")
            return {
                "query": query,
                "chunks": [],
                "warning": "No relevant medical guidelines found for this prediction."
            }

        print(f"Retrieved {len(chunks)} chunks (after dedup)\n")

        return {
            "query": query,
            "chunks": chunks
        }


if __name__ == "__main__":
    print("=== RAG Retriever Validation Test ===\n")

    retriever = RAGRetriever()

    # Normal case
    print("Test 1: Normal case — High risk + MI")
    result = retriever.retrieve_for_prediction("High", "MI", 35.0)
    for i, chunk in enumerate(result["chunks"]):
        print(f"  {i+1}. [{chunk['source']} p.{chunk['page']}] "
              f"confidence={chunk['confidence']}")

    # Edge case: no ECG class, no EF
    print("\nTest 2: Edge case — Medium risk only (no ECG, no EF)")
    result = retriever.retrieve_for_prediction("Medium")
    for i, chunk in enumerate(result["chunks"]):
        print(f"  {i+1}. [{chunk['source']} p.{chunk['page']}] "
              f"confidence={chunk['confidence']}")