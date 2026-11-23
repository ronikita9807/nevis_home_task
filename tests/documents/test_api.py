import uuid
from typing import Any

import pytest
from fastapi import status
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.documents.domain import MAX_CONTENT_LENGTH
from app.shared.settings import Settings
from tests.documents.fakes import FakeEmbedder
from tests.helpers import ADVISOR_A, ADVISOR_B, as_advisor

UTILITY_BILL: dict[str, Any] = {
    "title": "Utility bill — British Gas, March 2024",
    "content": "Energy statement for 14 Elm Road, Richmond. Amount due: £187.42.",
}


async def _create_client(http: AsyncClient, advisor: uuid.UUID) -> str:
    response = await http.post(
        "/clients",
        json={"first_name": "John", "last_name": "Doe", "email": "john.doe@neviswealth.com"},
        headers=as_advisor(advisor),
    )
    return str(response.json()["id"])


async def _chunk_rows(settings: Settings, document_id: str) -> list[Any]:
    engine = create_async_engine(str(settings.database_url), poolclass=NullPool)
    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                "SELECT chunk_index, advisor_id, embedding_model FROM document_chunks "
                "WHERE document_id = :id ORDER BY chunk_index"
            ),
            {"id": document_id},
        )
        rows = list(result)
    await engine.dispose()
    return rows


async def test_create_document_returns_201_spec_document_and_location(client: AsyncClient) -> None:
    client_id = await _create_client(client, ADVISOR_A)

    response = await client.post(
        f"/clients/{client_id}/documents", json=UTILITY_BILL, headers=as_advisor(ADVISOR_A)
    )

    assert response.status_code == status.HTTP_201_CREATED
    body = response.json()
    assert set(body) == {"id", "client_id", "title", "content", "created_at"}
    assert body["client_id"] == client_id
    assert response.headers["Location"].endswith(f"/documents/{body['id']}")
    fetched = await client.get(response.headers["Location"], headers=as_advisor(ADVISOR_A))
    assert fetched.json() == body


async def test_create_long_document_stores_chunks_owned_by_the_advisor(
    client: AsyncClient, settings: Settings
) -> None:
    client_id = await _create_client(client, ADVISOR_A)
    long_content = " ".join(f"Sentence number {i} about the client portfolio." for i in range(150))

    response = await client.post(
        f"/clients/{client_id}/documents",
        json={"title": "Annual review", "content": long_content},
        headers=as_advisor(ADVISOR_A),
    )

    rows = await _chunk_rows(settings, response.json()["id"])
    assert len(rows) > 1
    assert [row.chunk_index for row in rows] == list(range(len(rows)))
    assert {row.advisor_id for row in rows} == {ADVISOR_A}
    assert {row.embedding_model for row in rows} == {FakeEmbedder.model_name}


async def test_create_document_for_other_advisors_client_returns_404_without_inference(
    client: AsyncClient, fake_embedder: FakeEmbedder
) -> None:
    client_id = await _create_client(client, ADVISOR_A)

    response = await client.post(
        f"/clients/{client_id}/documents", json=UTILITY_BILL, headers=as_advisor(ADVISOR_B)
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["code"] == "client_not_found"
    assert fake_embedder.calls == 0


async def test_get_document_of_other_advisor_returns_404(client: AsyncClient) -> None:
    client_id = await _create_client(client, ADVISOR_A)
    created = await client.post(
        f"/clients/{client_id}/documents", json=UTILITY_BILL, headers=as_advisor(ADVISOR_A)
    )

    response = await client.get(created.headers["Location"], headers=as_advisor(ADVISOR_B))

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["code"] == "document_not_found"


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({**UTILITY_BILL, "title": "  "}, "title"),
        ({**UTILITY_BILL, "content": ""}, "content"),
        ({**UTILITY_BILL, "content": "x" * (MAX_CONTENT_LENGTH + 1)}, "content"),
        ({"title": "No content"}, "content"),
    ],
    ids=["blank-title", "empty-content", "content-too-long", "missing-content"],
)
async def test_create_document_invalid_payload_returns_422(
    client: AsyncClient, payload: dict[str, Any], field: str
) -> None:
    client_id = await _create_client(client, ADVISOR_A)

    response = await client.post(
        f"/clients/{client_id}/documents", json=payload, headers=as_advisor(ADVISOR_A)
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert field in {str(part) for error in response.json()["errors"] for part in error["loc"]}
