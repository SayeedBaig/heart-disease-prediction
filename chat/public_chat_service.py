"""
Public Chat Service
Author: Akash
Module 1 — Public AI Health Assistant

Purpose:
    Single entry point for the public chatbot (no login required).
    Orchestrates: safety check -> retrieval -> answer generation.
    This is what Sayeed's backend chat API will call.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever.retriever import RAGRetriever
from chat.safety_guardrails import check_query_safety
from chat.public_chat_generator import generate_public_answer


class PublicChatService:
    """
    Handles public (no-login) health education chatbot requests.
    Sayeed calls ask() and gets back the complete chatbot response.
    """

    def __init__(self):
        print("Initializing Public Chat Service...")
        self.retriever = RAGRetriever()
        print("Public Chat Service ready.\n")

    def ask(self, user_query: str) -> dict:
        """
        Full pipeline: safety check -> retrieve -> generate.

        Args:
            user_query: the patient's free-text question

        Returns:
            dict with answer, sources, recommend_doctor, status
        """
        try:
            # Step 1 — Safety check (block diagnostic/prediction requests)
            safety = check_query_safety(user_query)
            if not safety["safe"]:
                return {
                    "status": "blocked",
                    "query": user_query,
                    "answer": safety["redirect_message"],
                    "recommend_doctor": True,
                    "sources": []
                }

            # Step 2 — Retrieve relevant medical chunks
            retrieval = self.retriever.retrieve_for_query(user_query)
            chunks = retrieval["chunks"]

            if not chunks:
                return {
                    "status": "no_results",
                    "query": user_query,
                    "answer": (
                        "I don't have enough information to answer that "
                        "confidently. Please consult a healthcare "
                        "professional for guidance."
                    ),
                    "recommend_doctor": True,
                    "sources": []
                }

            # Step 3 — Generate conversational answer
            result = generate_public_answer(user_query, chunks)

            return {
                "status": "success",
                "query": user_query,
                "answer": result.get("answer", ""),
                "recommend_doctor": result.get("recommend_doctor", True),
                "sources": result.get("sources", [])
            }

        except Exception as e:
            print(f"Public Chat Service error: {e}")
            return {
                "status": "error",
                "query": user_query,
                "answer": (
                    "Something went wrong answering your question. "
                    "Please try again or consult a healthcare professional."
                ),
                "recommend_doctor": True,
                "sources": []
            }


if __name__ == "__main__":
    print("=== Public Chat Service Test ===\n")

    service = PublicChatService()

    test_questions = [
        "What causes chest pain?",
        "Do I have heart disease?",
        "What is ECG?",
        "How often should I exercise?",
    ]

    for q in test_questions:
        print(f"\n--- Q: {q} ---")
        result = service.ask(q)
        print(f"Status: {result['status']}")
        print(f"Answer: {result['answer']}")
        print(f"Recommend doctor: {result['recommend_doctor']}")
        print(f"Sources: {result['sources']}")