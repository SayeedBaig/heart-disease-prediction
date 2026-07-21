from fastapi import APIRouter

from api.schemas.rag import (
    ChatAskRequest,
    ChatAskResponse,
    RagHealthResponse,
)
from api.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])

# Module-level singleton so the FAISS index is loaded once at startup,
# not on every request.
_rag_service = RAGService()


@router.post(
    "/ask",
    summary="AI Health Assistant — ask a medical question",
    description=(
        "Public endpoint. Accepts a free-text medical question and an optional "
        "prediction context. Returns an LLM-generated answer grounded in "
        "retrieved cardiovascular medical guidelines, plus the source chunks "
        "used so the frontend can render citation cards.\n\n"
        "No authentication required."
    ),
    response_model=ChatAskResponse,
    response_description="Answer generated successfully.",
)
def ask(body: ChatAskRequest):
    """
    POST /rag/ask

    Public — no JWT required.
    """
    context_dict = body.context.model_dump() if body.context else None
    result = _rag_service.ask(question=body.question, context=context_dict)

    if result["status"] == "error":
        # Still return 200 with degraded payload so the frontend can show a
        # friendly fallback message rather than an HTTP error.
        return ChatAskResponse(
            answer=result["answer"],
            sources=[],
            status="error",
        )

    return ChatAskResponse(
        answer=result["answer"],
        sources=result["sources"],
        status="success",
    )


@router.get(
    "/health",
    summary="RAG subsystem health check",
    description=(
        "Returns the readiness state of the AI Health Assistant backend: "
        "whether the FAISS vector store is loaded, how many chunks are indexed, "
        "which LLM model is active, and whether the Groq API key is configured.\n\n"
        "No authentication required."
    ),
    response_model=RagHealthResponse,
)
def rag_health():
    """
    GET /rag/health

    Public — no JWT required.
    """
    return _rag_service.health()
