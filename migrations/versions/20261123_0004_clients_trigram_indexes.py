"""Trigram indexes on client fields for substring and typo-tolerant search.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-23
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COLUMNS = ("email", "first_name", "last_name", "description")


def upgrade() -> None:
    for column in COLUMNS:
        op.create_index(
            f"ix_clients_{column}_trgm",
            "clients",
            [column],
            postgresql_using="gin",
            postgresql_ops={column: "gin_trgm_ops"},
        )


def downgrade() -> None:
    for column in COLUMNS:
        op.drop_index(f"ix_clients_{column}_trgm", table_name="clients")
