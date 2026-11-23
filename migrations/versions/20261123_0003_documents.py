"""Create documents and their embedded chunks, scoped by advisor.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SEARCH_TSV_EXPRESSION = (
    "setweight(to_tsvector('english', title), 'A') || "
    "setweight(to_tsvector('english', content), 'B')"
)


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("client_id", sa.Uuid(), nullable=False),
        sa.Column("advisor_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "search_tsv",
            postgresql.TSVECTOR(),
            sa.Computed(SEARCH_TSV_EXPRESSION, persisted=True),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["client_id", "advisor_id"],
            ["clients.id", "clients.advisor_id"],
            name=op.f("fk_documents_client_id_advisor_id_clients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("id", "advisor_id", name=op.f("uq_documents_id_advisor_id")),
    )
    op.create_index(
        op.f("ix_documents_advisor_id_client_id"), "documents", ["advisor_id", "client_id"]
    )
    op.create_index(
        op.f("ix_documents_search_tsv"), "documents", ["search_tsv"], postgresql_using="gin"
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("advisor_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "advisor_id"],
            ["documents.id", "documents.advisor_id"],
            name=op.f("fk_document_chunks_document_id_advisor_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name=op.f("uq_document_chunks_document_id_chunk_index")
        ),
    )
    op.create_index(op.f("ix_document_chunks_advisor_id"), "document_chunks", ["advisor_id"])
    op.create_index(
        "ix_document_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("documents")
