"""Embeddings of client descriptions for search by meaning.

Clients created before this migration have no embedding until re-created (or a backfill
job embeds their descriptions); they remain searchable by keywords.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("clients", sa.Column("description_embedding", Vector(384), nullable=True))
    op.create_index(
        "ix_clients_description_embedding_hnsw",
        "clients",
        ["description_embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"description_embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_clients_description_embedding_hnsw", table_name="clients")
    op.drop_column("clients", "description_embedding")
