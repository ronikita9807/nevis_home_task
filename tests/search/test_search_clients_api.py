import uuid
from typing import Any

import pytest
from fastapi import status
from httpx import AsyncClient

from tests.helpers import ADVISOR_A, ADVISOR_B, as_advisor

CLIENTS: list[dict[str, Any]] = [
    {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@neviswealth.com",
        "description": "Senior software engineer, long-term growth investor",
    },
    {
        "first_name": "Emily",
        "last_name": "Carter",
        "email": "emily.carter@carterdesign.co.uk",
        "description": "Sold her design agency; cautious investor",
    },
    {"first_name": "Sophie", "last_name": "Müller", "email": "sophie.mueller@outlook.com"},
    {"first_name": "José", "last_name": "García", "email": "jose.garcia@garcia-family.es"},
    {"first_name": "Raj", "last_name": "Patel", "email": "raj_patel@gmail.com"},
]


@pytest.fixture
async def seeded(client: AsyncClient) -> AsyncClient:
    for payload in CLIENTS:
        response = await client.post("/clients", json=payload, headers=as_advisor(ADVISOR_A))
        assert response.status_code == status.HTTP_201_CREATED
    return client


async def _search(http: AsyncClient, q: str, advisor: uuid.UUID = ADVISOR_A) -> list[Any]:
    response = await http.get("/search", params={"q": q}, headers=as_advisor(advisor))
    assert response.status_code == status.HTTP_200_OK, response.text
    hits: list[Any] = response.json()
    return hits


def _emails(hits: list[Any]) -> list[str]:
    return [hit["client"]["email"] for hit in hits if hit["type"] == "client"]


async def test_search_company_name_finds_client_by_email_domain(seeded: AsyncClient) -> None:
    hits = await _search(seeded, "NevisWealth")

    assert _emails(hits) == ["john.doe@neviswealth.com"]
    assert hits[0]["type"] == "client"
    assert hits[0]["matched_fields"] == ["email"]


@pytest.mark.parametrize(
    ("query", "expected_email"),
    [
        ("doe john", "john.doe@neviswealth.com"),
        ("CARTER", "emily.carter@carterdesign.co.uk"),
        ("design agency", "emily.carter@carterdesign.co.uk"),
        ("müller", "sophie.mueller@outlook.com"),
        ("josé", "jose.garcia@garcia-family.es"),
    ],
    ids=["reversed-name", "case-insensitive", "description", "umlaut", "accent"],
)
async def test_search_finds_client_top_result(
    seeded: AsyncClient, query: str, expected_email: str
) -> None:
    hits = await _search(seeded, query)

    assert _emails(hits)[0] == expected_email


@pytest.mark.parametrize(
    ("query", "expected_email"),
    [("Jonh", "john.doe@neviswealth.com"), ("Cartre", "emily.carter@carterdesign.co.uk")],
)
async def test_search_tolerates_typos(seeded: AsyncClient, query: str, expected_email: str) -> None:
    assert expected_email in _emails(await _search(seeded, query))


async def test_search_ranks_exact_name_above_description_mention(seeded: AsyncClient) -> None:
    await seeded.post(
        "/clients",
        json={
            "first_name": "Anna",
            "last_name": "Lee",
            "email": "anna@lee.com",
            "description": "Referred by John Doe",
        },
        headers=as_advisor(ADVISOR_A),
    )

    emails = _emails(await _search(seeded, "John Doe"))

    assert emails[:2] == ["john.doe@neviswealth.com", "anna@lee.com"]


@pytest.mark.parametrize("query", ["%", "\\"])
async def test_search_like_wildcards_are_literal(seeded: AsyncClient, query: str) -> None:
    assert await _search(seeded, query) == []


@pytest.mark.parametrize("query", ["_", "raj_"])
async def test_search_underscore_matches_only_a_literal_underscore(
    seeded: AsyncClient, query: str
) -> None:
    assert _emails(await _search(seeded, query)) == ["raj_patel@gmail.com"]


async def test_search_never_returns_other_advisors_clients(seeded: AsyncClient) -> None:
    assert await _search(seeded, "NevisWealth", advisor=ADVISOR_B) == []


async def test_search_respects_limit_and_type_filter(seeded: AsyncClient) -> None:
    limited = await seeded.get(
        "/search", params={"q": "investor", "limit": 1}, headers=as_advisor(ADVISOR_A)
    )
    documents_only = await seeded.get(
        "/search", params={"q": "NevisWealth", "type": "document"}, headers=as_advisor(ADVISOR_A)
    )

    assert len(limited.json()) == 1
    assert documents_only.json() == []


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "   "},
        {"q": "x" * 257},
        {"q": "a", "limit": 0},
        {"q": "a", "type": "x"},
    ],
    ids=["missing", "empty", "blank", "too-long", "zero-limit", "unknown-type"],
)
async def test_search_invalid_parameters_return_422(
    client: AsyncClient, params: dict[str, Any]
) -> None:
    response = await client.get("/search", params=params, headers=as_advisor(ADVISOR_A))

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json()["code"] == "validation_error"
