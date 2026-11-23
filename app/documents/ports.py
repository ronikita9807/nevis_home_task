"""Ports of the documents module."""

import uuid
from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self

from app.documents.domain import Document, DocumentChunk, DocumentSummary, SummaryMethod


class Summarizer(Protocol):
    """Writes a short summary of a document. Raises `SummarizationError` when it cannot."""

    @property
    def method(self) -> SummaryMethod: ...

    async def summarize(self, title: str, content: str) -> str: ...


class ClientDirectory(Protocol):
    """What documents need to know about clients (implemented by the clients module)."""

    async def ensure_client_exists(self, advisor_id: uuid.UUID, client_id: uuid.UUID) -> None:
        """Raise `ClientNotFoundError` unless the advisor owns the client."""
        ...


class DocumentsRepository(Protocol):
    async def add(
        self, document: Document, chunks: Sequence[DocumentChunk], embedding_model: str
    ) -> None:
        """Persist a document with its chunks.

        Raises `ClientNotFoundError` if the client is gone or owned by another advisor.
        """
        ...

    async def get(self, advisor_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        """Return the advisor's document. Raises `DocumentNotFoundError` otherwise."""
        ...

    async def save_summary_if_absent(
        self, advisor_id: uuid.UUID, document_id: uuid.UUID, summary: DocumentSummary
    ) -> DocumentSummary:
        """Store the summary unless one exists; return the stored one (first writer wins)."""
        ...


class DocumentsUnitOfWork(Protocol):
    """One transaction: changes are applied by `commit()`, discarded on exit otherwise."""

    documents: DocumentsRepository

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...
