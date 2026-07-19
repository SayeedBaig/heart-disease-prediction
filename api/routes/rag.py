from fastapi import APIRouter

from api.schemas.rag_request import PublicChatRequest
from chat.public_chat_service import PublicChatService

router = APIRouter(prefix="/rag", tags=["RAG - AI Assistant"])

public_chat_service = PublicChatService()


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