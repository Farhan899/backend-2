"""Add priority and due_date columns to tasks table

Revision ID: 002
Revises: 001
Create Date: 2026-01-06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add priority column (nullable, defaults to NULL for existing rows)
    op.add_column('tasks',
        sa.Column('priority', sa.String(length=20), nullable=True)
    )

    # Add due_date column (nullable, defaults to NULL for existing rows)
    op.add_column('tasks',
        sa.Column('due_date', sa.DateTime(), nullable=True)
    )


def downgrade() -> None:
    # Remove columns in reverse order
    op.drop_column('tasks', 'due_date')
    op.drop_column('tasks', 'priority')
