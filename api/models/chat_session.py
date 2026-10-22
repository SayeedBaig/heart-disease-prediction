from datetime import datetime

from sqlalchemy import DateTime, String, Text, Index
from sqlalchemy.orm import Mapped, mapped_column

from api.database.base import Base


class ChatSession(Base):
    """Session-scoped report context; expires with the access token."""

    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_chat_sessions_expires_at", "expires_at"),)

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_role: Mapped[str] = mapped_column(String(16), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    report_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_embedding: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
