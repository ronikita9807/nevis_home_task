"""PostgreSQL implementation of the documents ports."""

import uuid
from collections.abc import Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Self

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.exceptions import ClientNotFoundError
from app.documents.domain import Document, DocumentChunk
from app.documents.exceptions import DocumentNotFoundError
from app.documents.infra.mappers import chunk_to_row, document_to_domain, document_to_row
from app.documents.infra.orm import DocumentRow
from app.shared.db import FOREIGN_KEY_VIOLATION, Database, violated_constraint

CLIENT_OWNERSHIP_FK = "fk_documents_client_id_advisor_id_clients"


class PostgreSQLDocumentsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self, document: Document, chunks: Sequence[DocumentChunk], embedding_model: str
    ) -> None:
        self._session.add(document_to_row(document))
        try:
            await self._session.flush()
        except IntegrityError as exc:
            if violated_constraint(exc, FOREIGN_KEY_VIOLATION) == CLIENT_OWNERSHIP_FK:
                raise ClientNotFoundError from exc
            raise
        # Chunks reference the document: inserted after it, in the same transaction.
        self._session.add_all(chunk_to_row(c, document, embedding_model) for c in chunks)
        await self._session.flush()

    async def get(self, advisor_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        row = await self._session.scalar(
            select(DocumentRow).where(
                DocumentRow.id == document_id, DocumentRow.visible_to(advisor_id)
            )
        )
        if row is None:
            raise DocumentNotFoundError
        return document_to_domain(row)


class PostgreSQLDocumentsUnitOfWork:
    documents: PostgreSQLDocumentsRepository

    def __init__(self, database: Database) -> None:
        self._database = database
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        self._session = await self._stack.enter_async_context(self._database.session())
        self.documents = PostgreSQLDocumentsRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._stack.__aexit__(exc_type, exc, traceback)

    async def commit(self) -> None:
        await self._session.commit()
