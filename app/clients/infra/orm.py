"""Storage model of clients."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ColumnElement,
    DateTime,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.db import Base
from app.shared.embeddings import EMBEDDING_DIMENSION


class ClientRow(Base):
    __tablename__ = "clients"
    __table_args__ = (
        # Uniqueness is per advisor: a global constraint would reveal, through 409,
        # that another advisor has a client with this email.
        UniqueConstraint("advisor_id", "email"),
        # Target of the composite FK from documents: a document's owner is its client's owner.
        UniqueConstraint("id", "advisor_id"),
        CheckConstraint("email = lower(email)", name="email_lowercase"),
        # Trigram indexes for search: substring (ILIKE) and typo-tolerant (<%) matching.
        *(
            Index(
                f"ix_clients_{column}_trgm",
                column,
                postgresql_using="gin",
                postgresql_ops={column: "gin_trgm_ops"},
            )
            for column in ("email", "first_name", "last_name", "description")
        ),
        Index(
            "ix_clients_description_embedding_hnsw",
            "description_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"description_embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    advisor_id: Mapped[uuid.UUID]
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    social_links: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Embedding of the description for search by meaning; NULL when there is no description.
    description_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSION))

    @classmethod
    def visible_to(cls, advisor_id: uuid.UUID) -> ColumnElement[bool]:
        """The single visibility rule for clients: an advisor sees only their own."""
        return cls.advisor_id == advisor_id
