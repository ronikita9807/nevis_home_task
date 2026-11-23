"""Storage model of clients."""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    ColumnElement,
    DateTime,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.db import Base


class ClientRow(Base):
    __tablename__ = "clients"
    __table_args__ = (
        # Uniqueness is per advisor: a global constraint would reveal, through 409,
        # that another advisor has a client with this email.
        UniqueConstraint("advisor_id", "email"),
        # Target of the composite FK from documents: a document's owner is its client's owner.
        UniqueConstraint("id", "advisor_id"),
        CheckConstraint("email = lower(email)", name="email_lowercase"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    advisor_id: Mapped[uuid.UUID]
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    social_links: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    def visible_to(cls, advisor_id: uuid.UUID) -> ColumnElement[bool]:
        """The single visibility rule for clients: an advisor sees only their own."""
        return cls.advisor_id == advisor_id
