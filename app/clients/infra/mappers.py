"""Explicit domain <-> storage mapping: a schema change breaks here, at type-check time."""

from app.clients.domain import Client
from app.clients.infra.orm import ClientRow


def client_to_row(client: Client, description_embedding: list[float] | None) -> ClientRow:
    return ClientRow(
        id=client.id,
        advisor_id=client.advisor_id,
        first_name=client.first_name,
        last_name=client.last_name,
        email=client.email,
        description=client.description,
        social_links=list(client.social_links),
        created_at=client.created_at,
        description_embedding=description_embedding,
    )


def client_to_domain(row: ClientRow) -> Client:
    return Client(
        id=row.id,
        advisor_id=row.advisor_id,
        first_name=row.first_name,
        last_name=row.last_name,
        email=row.email,
        description=row.description,
        social_links=list(row.social_links),
        created_at=row.created_at,
    )
