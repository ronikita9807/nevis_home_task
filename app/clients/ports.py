"""Ports of the clients module: what the service needs from the outside world."""

import uuid
from types import TracebackType
from typing import Protocol, Self

from app.clients.domain import Client


class ClientsRepository(Protocol):
    async def add(self, client: Client, description_embedding: list[float] | None) -> None:
        """Persist a new client (and the embedding of its description, if any).

        Raises `DuplicateClientEmailError` on a duplicate email.
        """
        ...

    async def get(self, advisor_id: uuid.UUID, client_id: uuid.UUID) -> Client:
        """Return the advisor's client. Raises `ClientNotFoundError` otherwise."""
        ...


class ClientsUnitOfWork(Protocol):
    """One transaction: changes are applied by `commit()`, discarded on exit otherwise."""

    clients: ClientsRepository

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...
