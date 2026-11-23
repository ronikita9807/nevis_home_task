"""Ports of the documents module."""

import uuid
from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self

from app.documents.domain import Document, DocumentChunk


class Embedder(Protocol):
    """Turns text into vectors whose cosine similarity reflects meaning.

    Queries and documents are embedded separately: retrieval models often expect a
    query-side instruction (prefix) that documents must not get.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


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
