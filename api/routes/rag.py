from fastapi import APIRouter

from api.schemas.rag_request import PublicChatRequest
from chat.public_chat_service import PublicChatService
from chat.suggested_questions import get_public_suggested_questions, get_doctor_suggested_questions
from chat.doctor_chat_service import DoctorChatService
from api.schemas.rag_request import PublicChatRequest, DoctorChatRequest
from chat.doctor_chat_service import DoctorChatService
from food_recommendation.food_service import FoodRecommendationService
from api.schemas.rag_request import PublicChatRequest, DoctorChatRequest, FoodRecommendationRequest
from chat.patient_chat_service import PatientChatService


from api.schemas.rag_request import (
    PublicChatRequest, DoctorChatRequest, FoodRecommendationRequest, PatientChatRequest
)


router = APIRouter(prefix="/rag", tags=["RAG - AI Assistant"])

public_chat_service = PublicChatService()
doctor_chat_service = DoctorChatService()
food_recommendation_service = FoodRecommendationService()
patient_chat_service = PatientChatService()

@router.post("/doctor/ask")
def ask_doctor_assistant(chat_request: DoctorChatRequest):
    """
    Doctor AI Assistant — requires doctor login (enforced by frontend/auth
    middleware, not this route directly).
    Evidence-based, clinical-terminology responses.
    Author: Akash
    """
    try:
        prediction_context = None
        if chat_request.risk_level or chat_request.ecg_class or chat_request.ef_value:
            prediction_context = {
                "risk_level": chat_request.risk_level,
                "ecg_class": chat_request.ecg_class,
                "ef_value": chat_request.ef_value
            }

        result = doctor_chat_service.ask(
            chat_request.question, prediction_context, chat_request.history
        )
        return {
            "success": True,
            **result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": "Something went wrong processing this request."
        }


@router.post("/public/ask")
def ask_public_assistant(chat_request: PublicChatRequest):
    """
    Public AI Health Assistant — no login required.
    Educates only. Never diagnoses or predicts.
    Author: Akash
    """
    try:
        result = public_chat_service.ask(chat_request.question)
        return {
            "success": True,
            **result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": (
                "Something went wrong answering your question. "
                "Please try again or consult a healthcare professional."
            )
        }
    


@router.get("/public/suggested-questions")
def suggested_questions():
    """
    Common questions shown on the public landing page.
    Author: Akash
    """
    return {
        "success": True,
        **get_public_suggested_questions()
    }

@router.post("/food-recommendation")
def get_food_recommendation(request: FoodRecommendationRequest):
    """
    Food & Lifestyle Recommendation Engine.
    Input: prediction risk level + patient profile flags.
    Output: recommended/avoid foods, water, exercise, sleep, lifestyle.
    Author: Akash
    """
    try:
        patient_profile = {
            "diabetes": request.diabetes,
            "smoker": request.smoker,
            "age": request.age
        }

        result = food_recommendation_service.get_recommendations(
            request.risk_level, patient_profile
        )

        return {
            "success": True,
            **result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/patient/ask")
def ask_patient_assistant(chat_request: PatientChatRequest):
    """
    Patient AI Assistant — simplified, jargon-free responses.
    risk_level (optional) enables food/lifestyle-aware answers.
    Author: Akash
    """
    try:
        result = patient_chat_service.ask(chat_request.question, chat_request.risk_level)
        return {
            "success": True,
            **result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": "Something went wrong. Please try again."
        }
    

@router.get("/doctor/suggested-questions")
def doctor_suggested_questions():
    """
    Suggested clinical questions for the doctor dashboard.
    Author: Akash
    """
    return {
        "success": True,
        **get_doctor_suggested_questions()
    }