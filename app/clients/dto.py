"""Read-side outputs of the clients module."""

import uuid
from typing import Self

from app.clients.domain import Client
from app.shared.domain import BaseDTO


class ClientDTO(BaseDTO):
    """`Client` schema from the spec. The owning advisor is not exposed."""

    id: uuid.UUID
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: list[str]

    @classmethod
    def from_domain(cls, client: Client) -> Self:
        return cls(
            id=client.id,
            first_name=client.first_name,
            last_name=client.last_name,
            email=client.email,
            description=client.description,
            social_links=list(client.social_links),
        )
