"""create doctors table

Revision ID: b31a3a1d4732
Revises: 22d60ec22e86
Create Date: 2026-07-16 23:12:40.515781

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b31a3a1d4732'
down_revision: Union[str, Sequence[str], None] = '22d60ec22e86'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'doctors',
        sa.Column('doctor_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('full_name', sa.String(100), nullable=False),
        sa.Column('specialization', sa.String(100), nullable=False),
        sa.Column('hospital', sa.String(150), nullable=False),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('doctors')
