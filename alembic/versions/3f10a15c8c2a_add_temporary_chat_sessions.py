"""add temporary report context and auth sessions

Revision ID: 3f10a15c8c2a
Revises: 85178fb28e85
Create Date: 2026-10-22
"""
from alembic import op
import sqlalchemy as sa

revision = "3f10a15c8c2a"
down_revision = "85178fb28e85"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("session_id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("user_role", sa.String(16), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("report_text", sa.Text(), nullable=True),
        sa.Column("report_embedding", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_index("ix_chat_sessions_expires_at", "chat_sessions", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_chat_sessions_expires_at", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_user_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")
