"""Cache document summaries and how they were produced.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("summary_method", sa.Text(), nullable=True))
    op.create_check_constraint(
        op.f("ck_documents_summary_method_known"),
        "documents",
        "summary_method IN ('llm', 'extractive')",
    )
    op.create_check_constraint(
        op.f("ck_documents_summary_with_method"),
        "documents",
        "(summary IS NULL) = (summary_method IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_documents_summary_with_method"), "documents", type_="check")
    op.drop_constraint(op.f("ck_documents_summary_method_known"), "documents", type_="check")
    op.drop_column("documents", "summary_method")
    op.drop_column("documents", "summary")
