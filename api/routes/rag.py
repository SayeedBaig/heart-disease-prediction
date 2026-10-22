from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from api.schemas.rag import (
    ChatAskRequest,
    ChatAskResponse,
    RagHealthResponse,
)
from api.services.rag_service import RAGService
from api.database.session import get_db
from api.utils.authorization import oauth2_scheme
from api.utils.auth import verify_access_token
from api.services.chat_session_service import active_session, delete_session, report_context

router = APIRouter(prefix="/rag", tags=["RAG"])

# Module-level singleton so the FAISS index is loaded once at startup,
# not on every request.
_rag_service = RAGService()
_optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/patients/login", auto_error=False)


@router.post(
    "/ask",
    summary="AI Health Assistant — ask a medical question",
    description=(
        "Accepts a free-text medical question and optional prediction context. "
        "Authenticated callers also receive their active session's latest report as context. "
        "Returns an LLM-generated answer grounded in "
        "retrieved cardiovascular medical guidelines, plus the source chunks "
        "used so the frontend can render citation cards.\n\n"
        "General questions remain available without authentication."
    ),
    response_model=ChatAskResponse,
    response_description="Answer generated successfully.",
)
def ask(
    body: ChatAskRequest,
    token: str | None = Depends(_optional_oauth2_scheme),
    db: Session = Depends(get_db),
):
    """
    POST /rag/ask

    Public for general questions. A valid access token enables session report context.
    """
    session = None
    if token:
        payload = verify_access_token(token)
        session = active_session(db, payload or {})
        if session is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid, expired, or ended session.")
    context_dict = body.context.model_dump() if body.context else None
    report_text = report_context(db, session, body.question) if session else None
    result = _rag_service.ask(question=body.question, context=context_dict, report_text=report_text)

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


@router.post("/logout", summary="End the authenticated assistant session")
def logout(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = verify_access_token(token)
    if payload is None:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
    delete_session(db, payload)
    return {"success": True}
