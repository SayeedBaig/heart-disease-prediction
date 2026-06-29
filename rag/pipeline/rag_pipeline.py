"""
RAG Pipeline
Author: Akash Hiremath 
Week: 4 — RAG Implementation (Core)

Purpose:
    Orchestrates the full RAG pipeline:
    Retriever → Generator → Structured Output
    This is the single entry point that Sayeed's RAGService will call.
"""

import os
import sys

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from rag.retriever.retriever import RAGRetriever, build_query
from rag.generator.generator import generate_explanation


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline orchestrator.
    Sayeed's RAGService calls run() with prediction data
    and receives a structured explanation back.
    """

    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.retriever = RAGRetriever()
        print("RAG Pipeline ready.\n")

    def run(self, prediction: dict) -> dict:
        """
        Run the full RAG pipeline for a given prediction.

        Args:
            prediction: dict containing:
                - risk_level: "Low", "Medium", or "High"
                - risk_percentage: int (0-100)
                - ecg_class: "NORM", "MI", "STTC", "CD", or "HYP"
                - ef_value: float or None

        Returns:
            dict containing:
                - query: the search query used
                - chunks: retrieved guideline chunks
                - explanation: generated medical explanation
                - status: "success" or "error"
        """
        try:
            risk_level = prediction.get("risk_level", "Medium")
            ecg_class = prediction.get("ecg_class", None)
            ef_value = prediction.get("ef_value", None)

            # Step 1 — Retrieve relevant chunks
            print(f"Step 1: Retrieving chunks for {risk_level} risk...")
            retrieval = self.retriever.retrieve_for_prediction(
                risk_level=risk_level,
                ecg_class=ecg_class,
                ef_value=ef_value
            )
            chunks = retrieval["chunks"]
            query = retrieval["query"]

            # Step 2 — Generate explanation
            print("Step 2: Generating explanation...")
            explanation = generate_explanation(prediction, chunks)

            return {
                "status": "success",
                "query": query,
                "chunks": chunks,
                "explanation": explanation
            }

        except Exception as e:
            print(f"RAG Pipeline error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": "",
                "chunks": [],
                "explanation": {
                    "summary": "Explanation unavailable.",
                    "details": "",
                    "recommendations": [],
                    "lifestyle_suggestions": []
                }
            }


# ── Entry Point (Test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RAG Pipeline Test ===\n")

    pipeline = RAGPipeline()

    # Test with High risk + MI
    test_prediction = {
        "risk_level": "High",
        "risk_percentage": 78,
        "ecg_class": "MI",
        "ef_value": 35.0
    }

    print(f"Input: {test_prediction}\n")
    result = pipeline.run(test_prediction)

    print(f"\nStatus: {result['status']}")
    print(f"Query: {result['query']}")
    print(f"Chunks retrieved: {len(result['chunks'])}")
    print("\nChunk sources:")
    for i, chunk in enumerate(result["chunks"]):
        print(f"  {i+1}. {chunk['source']} p.{chunk['page']}")

    print("\nExplanation:")
    print(f"  Summary: {result['explanation'].get('summary', 'N/A')}")
    recs = result['explanation'].get('recommendations', [])
    print(f"  Recommendations: {len(recs)}")