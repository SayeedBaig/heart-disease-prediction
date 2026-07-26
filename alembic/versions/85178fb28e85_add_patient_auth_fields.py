"""add patient auth fields

Revision ID: 85178fb28e85
Revises: a8c46f314c13
Create Date: 2026-07-21 18:47:27.248827

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '85178fb28e85'
down_revision: Union[str, Sequence[str], None] = 'a8c46f314c13'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Add password_hash temporarily as nullable
    op.add_column(
        "patients",
        sa.Column("password_hash", sa.String(length=255), nullable=True)
    )

    # Add is_active with a temporary default
    op.add_column(
        "patients",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true())
    )

    # Give existing patients a placeholder value
    op.execute("""
        UPDATE patients
        SET password_hash = 'TEMP_PASSWORD_RESET_REQUIRED'
        WHERE password_hash IS NULL
    """)

    # Make password_hash NOT NULL
    op.alter_column(
        "patients",
        "password_hash",
        nullable=False
    )

    # Remove the default from is_active
    op.alter_column(
        "patients",
        "is_active",
        server_default=None
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_column("patients", "is_active")
    op.drop_column("patients", "password_hash")