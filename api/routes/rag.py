from fastapi import APIRouter

from api.schemas.rag_request import PublicChatRequest
from chat.public_chat_service import PublicChatService
from chat.suggested_questions import get_public_suggested_questions
from chat.doctor_chat_service import DoctorChatService
from api.schemas.rag_request import PublicChatRequest, DoctorChatRequest


router = APIRouter(prefix="/rag", tags=["RAG - AI Assistant"])

public_chat_service = PublicChatService()
doctor_chat_service = DoctorChatService()

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

        result = doctor_chat_service.ask(chat_request.question, prediction_context)
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