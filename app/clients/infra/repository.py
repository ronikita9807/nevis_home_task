"""PostgreSQL implementation of the clients ports."""

import uuid
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Self

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.domain import Client
from app.clients.exceptions import ClientNotFoundError, DuplicateClientEmailError
from app.clients.infra.mappers import client_to_domain, client_to_row
from app.clients.infra.orm import ClientRow
from app.shared.db import Database, is_unique_violation

UNIQUE_EMAIL_CONSTRAINT = "uq_clients_advisor_id_email"


class PostgreSQLClientsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, client: Client, description_embedding: list[float] | None) -> None:
        self._session.add(client_to_row(client, description_embedding))
        try:
            # Flush now so a duplicate surfaces here, as a domain error, not at commit.
            await self._session.flush()
        except IntegrityError as exc:
            if is_unique_violation(exc, UNIQUE_EMAIL_CONSTRAINT):
                raise DuplicateClientEmailError from exc
            raise

    async def get(self, advisor_id: uuid.UUID, client_id: uuid.UUID) -> Client:
        row = await self._session.scalar(
            select(ClientRow).where(ClientRow.id == client_id, ClientRow.visible_to(advisor_id))
        )
        if row is None:
            raise ClientNotFoundError
        return client_to_domain(row)


class PostgreSQLClientsUnitOfWork:
    clients: PostgreSQLClientsRepository

    def __init__(self, database: Database) -> None:
        self._database = database
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        self._session = await self._stack.enter_async_context(self._database.session())
        self.clients = PostgreSQLClientsRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Closing the session rolls back anything not committed.
        await self._stack.__aexit__(exc_type, exc, traceback)

    async def commit(self) -> None:
        await self._session.commit()
