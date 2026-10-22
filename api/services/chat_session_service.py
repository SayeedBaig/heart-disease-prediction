from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
from jose import jwt
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from api.models.chat_session import ChatSession
from api.utils.auth import ALGORITHM, SECRET_KEY

logger = logging.getLogger(__name__)
_embedding_model = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def register_token_session(db: Session, token: str) -> None:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    db.add(ChatSession(
        session_id=payload["jti"], user_id=str(payload["sub"]),
        user_role=payload["role"], expires_at=expires_at, updated_at=_now(),
    ))
    db.execute(delete(ChatSession).where(ChatSession.expires_at <= _now()).execution_options(synchronize_session=False))
    db.commit()


def active_session(db: Session, payload: dict) -> ChatSession | None:
    session_id = payload.get("jti")
    if not session_id:
        return None
    db.execute(delete(ChatSession).where(ChatSession.expires_at <= _now()).execution_options(synchronize_session=False))
    row = db.get(ChatSession, session_id)
    if (row is None or _utc(row.expires_at) <= _now() or row.user_id != str(payload.get("sub"))
            or row.user_role != payload.get("role")):
        db.commit()
        return None
    db.commit()
    return row


def delete_session(db: Session, payload: dict) -> None:
    if payload.get("jti"):
        db.execute(delete(ChatSession).where(ChatSession.session_id == payload["jti"]).execution_options(synchronize_session=False))
    db.execute(delete(ChatSession).where(ChatSession.expires_at <= _now()).execution_options(synchronize_session=False))
    db.commit()


def _embed(text: str) -> list[float]:
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        from rag.model_config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION)
    return _embedding_model.encode([text], normalize_embeddings=True)[0].astype("float32").tolist()


def replace_report(db: Session, session: ChatSession, report: dict) -> None:
    """Prepare embedding first, then replace the prior report atomically."""
    report_text = _report_to_text(report)
    embedding = json.dumps(_embed(report_text))
    current = db.scalar(select(ChatSession).where(ChatSession.session_id == session.session_id).with_for_update())
    if current is None or _utc(current.expires_at) <= _now():
        raise ValueError("Authenticated session has expired.")
    current.report_text = report_text
    current.report_embedding = embedding
    current.updated_at = _now()
    db.commit()
    logger.info("Updated temporary report context for active chat session")


def _report_to_text(report: dict) -> str:
    # Keep only user-facing clinical report content; omit identifying details.
    patient = report.get("patient") or {}
    sanitized = {key: value for key, value in report.items() if key != "patient"}
    if patient.get("gender"):
        sanitized["patient_gender"] = patient["gender"]
    return json.dumps(sanitized, ensure_ascii=False, default=str)


def report_context(db: Session, session: ChatSession, query: str) -> str | None:
    current = db.get(ChatSession, session.session_id)
    if not current or not current.report_text or not current.report_embedding:
        return None
    query_embedding = np.asarray(_embed(query), dtype="float32")
    report_embedding = np.asarray(json.loads(current.report_embedding), dtype="float32")
    # One report is active per token session. Include it whenever present so
    # retrieval cannot rank an older report above the latest one.
    if query_embedding.shape != report_embedding.shape:
        return None
    return current.report_text
