"""Read-side outputs of the documents module."""

import uuid
from datetime import datetime
from typing import Self

from app.documents.domain import Document, DocumentSummary, SummaryMethod
from app.shared.domain import BaseDTO


class DocumentDTO(BaseDTO):
    """`Document` schema from the spec."""

    id: uuid.UUID
    client_id: uuid.UUID
    title: str
    content: str
    created_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> Self:
        return cls(
            id=document.id,
            client_id=document.client_id,
            title=document.title,
            content=document.content,
            created_at=document.created_at,
        )


class SummaryDTO(BaseDTO):
    """Short summary of a document and how it was produced."""

    document_id: uuid.UUID
    summary: str
    method: SummaryMethod

    @classmethod
    def build(cls, document_id: uuid.UUID, summary: DocumentSummary) -> Self:
        return cls(document_id=document_id, summary=summary.text, method=summary.method)
