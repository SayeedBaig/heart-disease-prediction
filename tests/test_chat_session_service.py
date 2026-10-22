import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.database.base import Base
from api.models.chat_session import ChatSession
from api.services import chat_session_service as sessions
from rag.generator.generator import build_chat_prompt


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[ChatSession.__table__])
    with Session(engine) as session:
        yield session


def make_session(db, session_id="session-a", user_id="patient-1", role="patient", expires=None):
    row = ChatSession(
        session_id=session_id, user_id=user_id, user_role=role,
        expires_at=expires or datetime.now(timezone.utc) + timedelta(minutes=30),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    return row


def test_report_is_available_only_in_its_active_session(db, monkeypatch):
    first = make_session(db)
    second = make_session(db, session_id="session-b", user_id="patient-2")
    monkeypatch.setattr(sessions, "_embed", lambda _text: [1.0, 0.0])

    sessions.replace_report(db, first, {"summary": "Latest patient report"})

    assert "Latest patient report" in sessions.report_context(db, first, "What is my risk?")
    assert sessions.report_context(db, second, "What is my risk?") is None
    sessions.replace_report(db, first, {"summary": "Replacement report"})
    assert "Replacement report" in sessions.report_context(db, first, "What is my risk?")
    assert "Latest patient report" not in first.report_text


def test_failed_embedding_preserves_previous_report(db, monkeypatch):
    row = make_session(db)
    monkeypatch.setattr(sessions, "_embed", lambda _text: [1.0, 0.0])
    sessions.replace_report(db, row, {"summary": "Valid report"})
    monkeypatch.setattr(sessions, "_embed", lambda _text: (_ for _ in ()).throw(RuntimeError("failed")))

    with pytest.raises(RuntimeError):
        sessions.replace_report(db, row, {"summary": "Broken replacement"})

    db.refresh(row)
    assert "Valid report" in row.report_text
    assert json.loads(row.report_embedding) == [1.0, 0.0]


def test_logout_removes_report_and_expiration_invalidates_session(db):
    row = make_session(db)
    row.report_text = "Temporary"
    row.report_embedding = "[]"
    db.commit()

    session_id = row.session_id
    sessions.delete_session(db, {"jti": session_id})
    assert db.get(ChatSession, session_id) is None

    expired = make_session(db, session_id="expired", expires=datetime.now(timezone.utc) - timedelta(seconds=1))
    expired_payload = {"jti": expired.session_id, "sub": expired.user_id, "role": expired.user_role}
    assert sessions.active_session(db, expired_payload) is None
    assert db.get(ChatSession, expired_payload["jti"]) is None


def test_chat_prompt_uses_report_as_separate_current_session_context():
    prompt = build_chat_prompt("What does my report say?", [
        {"text": "risk percentage: 40", "source": "Your latest session report", "page": 1, "category": "patient_report"},
        {"text": "Guideline advice", "source": "AHA", "page": 3, "category": "risk_factors"},
    ])
    assert "LATEST REPORT FOR THE CURRENT AUTHENTICATED SESSION" in prompt
    assert "risk percentage: 40" in prompt
    assert "Guideline advice" in prompt
