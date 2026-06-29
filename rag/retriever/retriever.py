"""
RAG Retriever
Author: Akash Hiremath (1RF23CS013)
Week: 3 — RAG Implementation

Purpose:
    Builds search queries from fusion output and retrieves
    the most relevant medical guideline chunks from FAISS index.
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

VECTOR_STORE_DIR = "rag/vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

# ── Query Builder ─────────────────────────────────────────────────────────────

def build_query(risk_level: str, ecg_class: str = None,
                ef_value: float = None) -> str:
    """
    Build a clinically meaningful search query from fusion output.

    Args:
        risk_level: "Low", "Medium", or "High"
        ecg_class: "NORM", "MI", "STTC", "CD", or "HYP"
        ef_value: Ejection fraction percentage

    Returns:
        A search query string for FAISS retrieval
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


# ── Retriever ─────────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Retrieves relevant medical guideline chunks from FAISS index
    based on a search query built from fusion output.
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
        Retrieve top-k relevant chunks from FAISS index.

        Args:
            query: Search query string from build_query()
            top_k: Number of chunks to return

        Returns:
            List of dicts with text, source, category, page
        """
        embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "category": self.metadata[idx]["category"],
                "page": self.metadata[idx]["page"],
                "score": float(distances[0][i])
            })

        return results

    def retrieve_for_prediction(self, risk_level: str,
                                 ecg_class: str = None,
                                 ef_value: float = None) -> dict:
        """
        Full pipeline: build query from fusion output → retrieve chunks.

        Args:
            risk_level: "Low", "Medium", or "High"
            ecg_class: ECG classification string
            ef_value: Ejection fraction value

        Returns:
            dict with query and retrieved chunks
        """
        query = build_query(risk_level, ecg_class, ef_value)
        print(f"Query: {query}")

        chunks = self.retrieve(query)
        print(f"Retrieved {len(chunks)} chunks\n")

        return {
            "query": query,
            "chunks": chunks
        }


# ── Entry Point (Test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RAG Retriever Test ===\n")

    retriever = RAGRetriever()

    # Test Case 1 — High risk + MI
    print("Test 1: High risk + MI")
    result = retriever.retrieve_for_prediction(
        risk_level="High",
        ecg_class="MI",
        ef_value=35.0
    )
    for i, chunk in enumerate(result["chunks"]):
        print(f"  Chunk {i+1}: [{chunk['source']} p.{chunk['page']}]")
        print(f"    {chunk['text'][:100]}...")

    print("\nTest 2: Low risk + NORM")
    result = retriever.retrieve_for_prediction(
        risk_level="Low",
        ecg_class="NORM",
        ef_value=60.0
    )
    for i, chunk in enumerate(result["chunks"]):
        print(f"  Chunk {i+1}: [{chunk['source']} p.{chunk['page']}]")
        print(f"    {chunk['text'][:100]}...")