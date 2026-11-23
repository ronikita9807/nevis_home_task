"""Storage model of documents and their embedded chunks."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ColumnElement,
    Computed,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.db import Base
from app.shared.embeddings import EMBEDDING_DIMENSION

# English stemming: the client base is UK/US (an assumption, see FastEmbedEmbedder).
# Multilingual search would switch this to 'simple' (new migration of the generated column).
FTS_CONFIG = "english"
SEARCH_TSV_EXPRESSION = (
    f"setweight(to_tsvector('{FTS_CONFIG}', title), 'A') || "
    f"setweight(to_tsvector('{FTS_CONFIG}', content), 'B')"
)


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # The owner is copied from the client and the database keeps them equal:
        # a document can only reference a client of the same advisor.
        ForeignKeyConstraint(
            ["client_id", "advisor_id"], ["clients.id", "clients.advisor_id"], ondelete="CASCADE"
        ),
        UniqueConstraint("id", "advisor_id"),  # target of the chunks' composite FK
        Index(None, "advisor_id", "client_id"),
        Index(None, "search_tsv", postgresql_using="gin"),
        CheckConstraint("summary_method IN ('llm', 'extractive')", name="summary_method_known"),
        CheckConstraint("(summary IS NULL) = (summary_method IS NULL)", name="summary_with_method"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    client_id: Mapped[uuid.UUID]
    advisor_id: Mapped[uuid.UUID]
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    search_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed(SEARCH_TSV_EXPRESSION, persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    summary: Mapped[str | None] = mapped_column(Text)
    summary_method: Mapped[str | None] = mapped_column(Text)

    @classmethod
    def visible_to(cls, advisor_id: uuid.UUID) -> ColumnElement[bool]:
        return cls.advisor_id == advisor_id


class DocumentChunkRow(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "advisor_id"],
            ["documents.id", "documents.advisor_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("document_id", "chunk_index"),
        Index(None, "advisor_id"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID]
    # Denormalised owner: the vector search filters by it without joining documents.
    advisor_id: Mapped[uuid.UUID]
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION))
    embedding_model: Mapped[str] = mapped_column(Text)

    @classmethod
    def visible_to(cls, advisor_id: uuid.UUID) -> ColumnElement[bool]:
        return cls.advisor_id == advisor_id
