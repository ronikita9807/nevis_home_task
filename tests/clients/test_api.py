import asyncio
import uuid
from typing import Any

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.main import create_app
from app.shared.settings import Settings
from tests.helpers import ADVISOR_A, ADVISOR_B, as_advisor

JOHN: dict[str, Any] = {
    "first_name": "John",
    "last_name": "Doe",
    "email": "John.Doe@NevisWealth.com",
    "description": "Long-term growth investor",
    "social_links": ["https://www.linkedin.com/in/john-doe-example"],
}


async def test_create_client_returns_201_with_location_and_normalized_email(
    client: AsyncClient,
) -> None:
    response = await client.post("/clients", json=JOHN, headers=as_advisor(ADVISOR_A))

    assert response.status_code == status.HTTP_201_CREATED
    body = response.json()
    assert body["email"] == "john.doe@neviswealth.com"
    assert body["first_name"] == "John"
    assert response.headers["Location"].endswith(f"/clients/{body['id']}")
    fetched = await client.get(response.headers["Location"], headers=as_advisor(ADVISOR_A))
    assert fetched.json() == body


async def test_create_client_with_only_required_fields_defaults_optional_ones(
    client: AsyncClient,
) -> None:
    payload = {"first_name": "Zoë", "last_name": "O'Brien", "email": "zoe@example.com"}

    response = await client.post("/clients", json=payload, headers=as_advisor(ADVISOR_A))

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["description"] is None
    assert response.json()["social_links"] == []


async def test_create_client_duplicate_email_for_same_advisor_returns_409(
    client: AsyncClient,
) -> None:
    await client.post("/clients", json=JOHN, headers=as_advisor(ADVISOR_A))
    duplicate = {**JOHN, "email": "john.doe@neviswealth.com"}  # differs only in case

    response = await client.post("/clients", json=duplicate, headers=as_advisor(ADVISOR_A))

    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["code"] == "duplicate_client_email"


async def test_create_client_same_email_for_other_advisor_succeeds(client: AsyncClient) -> None:
    first = await client.post("/clients", json=JOHN, headers=as_advisor(ADVISOR_A))
    second = await client.post("/clients", json=JOHN, headers=as_advisor(ADVISOR_B))

    assert first.status_code == status.HTTP_201_CREATED
    assert second.status_code == status.HTTP_201_CREATED
    assert first.json()["id"] != second.json()["id"]


async def test_create_client_concurrent_duplicates_create_exactly_one(client: AsyncClient) -> None:
    responses = await asyncio.gather(
        *(client.post("/clients", json=JOHN, headers=as_advisor(ADVISOR_A)) for _ in range(5))
    )

    codes = sorted(r.status_code for r in responses)
    assert codes == [status.HTTP_201_CREATED] + [status.HTTP_409_CONFLICT] * 4


async def test_get_client_of_other_advisor_returns_404(client: AsyncClient) -> None:
    created = await client.post("/clients", json=JOHN, headers=as_advisor(ADVISOR_A))

    response = await client.get(f"/clients/{created.json()['id']}", headers=as_advisor(ADVISOR_B))

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["code"] == "client_not_found"


async def test_get_unknown_client_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/clients/{uuid.uuid4()}", headers=as_advisor(ADVISOR_A))

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({**JOHN, "email": "not-an-email"}, "email"),
        ({**JOHN, "first_name": "   "}, "first_name"),
        ({k: v for k, v in JOHN.items() if k != "last_name"}, "last_name"),
        ({**JOHN, "social_links": ["x"] * 11}, "social_links"),
        ({**JOHN, "advisor_id": str(ADVISOR_B)}, "advisor_id"),
    ],
    ids=["invalid-email", "blank-name", "missing-last-name", "too-many-links", "advisor-in-body"],
)
async def test_create_client_invalid_payload_returns_422_problem(
    client: AsyncClient, payload: dict[str, Any], field: str
) -> None:
    response = await client.post("/clients", json=payload, headers=as_advisor(ADVISOR_A))

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert field in {str(part) for error in response.json()["errors"] for part in error["loc"]}


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Advisor-Id": "not-a-uuid"}],
    ids=["missing", "malformed"],
)
async def test_client_endpoints_without_valid_advisor_return_401(
    client: AsyncClient, headers: dict[str, str]
) -> None:
    response = await client.post("/clients", json=JOHN, headers=headers)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["code"] == "unauthorized"


@pytest.mark.parametrize(
    ("key", "expected"),
    [(None, 401), ("wrong", 401), ("s3cret", 201)],
    ids=["missing", "wrong", "valid"],
)
async def test_service_key_when_configured_is_enforced(
    settings: Settings, key: str | None, expected: int
) -> None:
    app = create_app(settings.model_copy(update={"api_key": SecretStr("s3cret")}))
    headers = as_advisor(ADVISOR_A) | ({"X-API-Key": key} if key else {})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post("/clients", json=JOHN, headers=headers)
    await app.state.container.database().dispose()

    assert response.status_code == expected
