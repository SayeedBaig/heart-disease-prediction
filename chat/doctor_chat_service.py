"""
Doctor Chat Service
Author: Akash
Module 2 — Doctor AI Assistant

Purpose:
    Entry point for the Doctor AI Assistant (post-login only).
    Unlike Public chat, this does NOT run the diagnostic-block
    guardrail — a doctor discussing their patient's case is
    legitimate clinical use, not a diagnosis request.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever.retriever import RAGRetriever
from chat.doctor_chat_generator import generate_doctor_answer


class DoctorChatService:
    """
    Handles Doctor AI Assistant requests (requires doctor login).
    prediction_context is optional — passed when the doctor is
    asking about a specific patient's case.
    """

    def __init__(self):
        print("Initializing Doctor Chat Service...")
        self.retriever = RAGRetriever()
        print("Doctor Chat Service ready.\n")

    def ask(self, user_query: str, prediction_context: dict = None, history: list = None) -> dict:
        try:
            # Build retrieval query — if there's prediction context,
            # blend it into the search so retrieval is more targeted
            retrieval_query = user_query
            if prediction_context:
                risk = prediction_context.get("risk_level", "")
                ecg = prediction_context.get("ecg_class", "")
                retrieval_query = f"{user_query} {risk} risk {ecg}".strip()

            retrieval = self.retriever.retrieve_for_query(retrieval_query)
            chunks = retrieval["chunks"]

            if not chunks:
                return {
                    "status": "no_results",
                    "query": user_query,
                    "answer": (
                        "No relevant clinical guidelines found for this "
                        "question. Please rephrase or consult primary "
                        "literature directly."
                    ),
                    "clinical_guidelines_cited": [],
                    "recommended_next_steps": []
                }

            result = generate_doctor_answer(user_query, chunks, prediction_context, history)

            return {
                "status": "success",
                "query": user_query,
                "answer": result.get("answer", ""),
                "clinical_guidelines_cited": result.get("clinical_guidelines_cited", []),
                "recommended_next_steps": result.get("recommended_next_steps", [])
            }

        except Exception as e:
            print(f"Doctor Chat Service error: {e}")
            return {
                "status": "error",
                "query": user_query,
                "answer": "Something went wrong processing this request.",
                "clinical_guidelines_cited": [],
                "recommended_next_steps": []
            }


if __name__ == "__main__":
    print("=== Doctor Chat Service Test ===\n")

    service = DoctorChatService()

    print("\n--- Test 1: General clinical question, no case context ---")
    result = service.ask("What are the recommended diagnostic tests for suspected HFrEF?")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}")
    print(f"Guidelines: {result['clinical_guidelines_cited']}")
    print(f"Next steps: {result['recommended_next_steps']}")

    print("\n--- Test 2: Question tied to a specific patient's prediction ---")
    prediction_context = {"risk_level": "High", "ecg_class": "MI", "ef_value": 35.0}
    result = service.ask("Why is this prediction High Risk?", prediction_context)
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}")
    print(f"Guidelines: {result['clinical_guidelines_cited']}")
    print(f"Next steps: {result['recommended_next_steps']}")

    print("\n--- Test 3: History comparison question ---")
    history = [
        {"date": "2026-05-01", "risk_level": "Medium", "ecg_class": "Normal", "ef_value": 55.0},
        {"date": "2026-07-01", "risk_level": "High", "ecg_class": "MI", "ef_value": 35.0}
    ]
    result = service.ask("Explain why risk increased compared to previous visit.", history=history)
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}")