"""
Patient Chat Service
Author: Akash
Module 3 — Patient AI Assistant
Module 6 — Context Awareness

Purpose:
    Entry point for the Patient AI Assistant. Reuses the same
    safety guardrail as Module 1 (patients shouldn't get diagnoses
    either). Pulls food/lifestyle context from Module 4 when the
    patient has a known risk level. Uses conversation memory to
    handle follow-up questions naturally.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever.retriever import RAGRetriever
from chat.safety_guardrails import check_query_safety
from chat.patient_chat_generator import generate_patient_answer
from food_recommendation.food_service import FoodRecommendationService
from chat.conversation_memory import build_contextual_query, add_turn, get_history


class PatientChatService:
    """
    Handles Patient AI Assistant requests (post-login, patient view).
    risk_level is optional — passed when we know the patient's
    prediction result, so food/lifestyle guidance can be included.
    session_id is optional — passed to enable conversation memory
    for follow-up questions.
    """

    def __init__(self):
        print("Initializing Patient Chat Service...")
        self.retriever = RAGRetriever()
        self.food_service = FoodRecommendationService()
        print("Patient Chat Service ready.\n")

    def ask(self, user_query: str, risk_level: str = None, session_id: str = None) -> dict:
        try:
            # Step 1 — Safety check (same guardrail as public chatbot)
            safety = check_query_safety(user_query)
            if not safety["safe"]:
                return {
                    "status": "blocked",
                    "query": user_query,
                    "answer": safety["redirect_message"],
                    "recommend_doctor": True
                }

            # Step 2 — Retrieve relevant medical chunks (context-aware if session_id given)
            retrieval_query = user_query
            if session_id:
                retrieval_query = build_contextual_query(session_id, user_query)

            retrieval = self.retriever.retrieve_for_query(retrieval_query)
            chunks = retrieval["chunks"]

            if not chunks:
                return {
                    "status": "no_results",
                    "query": user_query,
                    "answer": (
                        "I don't have enough information to answer that. "
                        "Please ask your doctor about this."
                    ),
                    "recommend_doctor": True
                }

            # Step 3 — Pull food/lifestyle context if risk level is known
            food_context = None
            if risk_level:
                food_result = self.food_service.get_recommendations(risk_level)
                if food_result["status"] == "success":
                    food_context = {
                        "recommended_foods": food_result["recommended_foods"],
                        "avoid_foods": food_result["avoid_foods"],
                        "exercise": food_result["exercise"]
                    }

            # Step 4 — Generate simplified answer, passing conversation
            # history so follow-ups are answered directly, not generically
            conversation_history = get_history(session_id) if session_id else None
            result = generate_patient_answer(user_query, chunks, food_context, conversation_history)

            if session_id:
                add_turn(session_id, user_query, result.get("answer", ""))

            return {
                "status": "success",
                "query": user_query,
                "answer": result.get("answer", ""),
                "recommend_doctor": result.get("recommend_doctor", True)
            }

        except Exception as e:
            print(f"Patient Chat Service error: {e}")
            return {
                "status": "error",
                "query": user_query,
                "answer": (
                    "Something went wrong. Please try again or ask "
                    "your doctor."
                ),
                "recommend_doctor": True
            }


if __name__ == "__main__":
    print("=== Patient Chat Service Test ===\n")

    service = PatientChatService()
    session = "test-session"

    print("\n--- Turn 1: Can I exercise? ---")
    result = service.ask("Can I exercise?", risk_level="Medium", session_id=session)
    print(f"Answer: {result['answer']}")

    print("\n--- Turn 2: How long? (follow-up, should use context) ---")
    result = service.ask("How long?", risk_level="Medium", session_id=session)
    print(f"Answer: {result['answer']}")