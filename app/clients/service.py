"""Use cases of the clients module."""

import uuid
from collections.abc import Callable

from app.clients.commands import CreateClientCommand
from app.clients.domain import Client
from app.clients.dto import ClientDTO
from app.clients.ports import ClientsUnitOfWork
from app.shared.embeddings import Embedder


class ClientsService:
    def __init__(self, uow_factory: Callable[[], ClientsUnitOfWork], embedder: Embedder) -> None:
        self._uow_factory = uow_factory
        self._embedder = embedder

    async def create_client(self, advisor_id: uuid.UUID, command: CreateClientCommand) -> ClientDTO:
        client = Client.create(
            advisor_id=advisor_id,
            first_name=command.first_name,
            last_name=command.last_name,
            email=command.email,
            description=command.description,
            social_links=list(command.social_links),
        )
        # Embedded before the transaction, as for documents: no connection held during inference.
        embedding = (
            (await self._embedder.embed_documents([client.description]))[0]
            if client.description
            else None
        )
        async with self._uow_factory() as uow:
            await uow.clients.add(client, embedding)
            await uow.commit()
        return ClientDTO.from_domain(client)

    async def get_client(self, advisor_id: uuid.UUID, client_id: uuid.UUID) -> ClientDTO:
        async with self._uow_factory() as uow:
            client = await uow.clients.get(advisor_id, client_id)
        return ClientDTO.from_domain(client)

    async def ensure_client_exists(self, advisor_id: uuid.UUID, client_id: uuid.UUID) -> None:
        """Raise `ClientNotFoundError` unless the advisor owns the client."""
        async with self._uow_factory() as uow:
            await uow.clients.get(advisor_id, client_id)
