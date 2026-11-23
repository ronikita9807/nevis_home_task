"""Document search through the API with the fake embedder: keyword + thesaurus channel,
response shape and isolation. Real semantics are covered by the golden set."""

from typing import Any

import pytest
from fastapi import status
from httpx import AsyncClient

from tests.helpers import ADVISOR_A, ADVISOR_B, as_advisor
from tests.search.corpus import seed_demo_corpus


@pytest.fixture
async def documents(client: AsyncClient) -> dict[str, str]:
    return await seed_demo_corpus(client)


async def _search(http: AsyncClient, **params: Any) -> list[dict[str, Any]]:
    advisor = params.pop("advisor", ADVISOR_A)
    response = await http.get("/search", params=params, headers=as_advisor(advisor))
    assert response.status_code == status.HTTP_200_OK, response.text
    hits: list[dict[str, Any]] = response.json()
    return hits


async def test_address_proof_finds_documents_containing_utility_bill(
    client: AsyncClient, documents: dict[str, str]
) -> None:
    await client.post(
        f"/clients/{(await _first_client_id(client))}/documents",
        json={"title": "Utility bill", "content": "Thames Water utility bill for March."},
        headers=as_advisor(ADVISOR_A),
    )

    hits = await _search(client, q="address proof", type="document")

    titles = [hit["document"]["title"] for hit in hits]
    assert "Utility bill" in titles[:3]
    utility = next(hit for hit in hits if hit["document"]["title"] == "Utility bill")
    assert "keyword" in utility["matched_by"]
    assert "utility bill" in utility["snippet"].lower()


async def test_document_hit_has_spec_summary_snippet_and_client(
    client: AsyncClient, documents: dict[str, str]
) -> None:
    hits = await _search(client, q="council tax", type="document")

    hit = hits[0]
    assert hit["type"] == "document"
    assert hit["document"]["id"] == documents["emily_council_tax"]
    assert set(hit["document"]) == {"id", "client_id", "title", "created_at"}
    assert hit["client"]["last_name"] == "Carter"
    assert hit["snippet"]
    assert hit["score"] > 0


async def test_search_returns_client_block_before_document_block(
    client: AsyncClient, documents: dict[str, str]
) -> None:
    hits = await _search(client, q="Carter")

    types = [hit["type"] for hit in hits]
    assert types[0] == "client"
    assert "document" in types
    assert types == sorted(types)  # "client" < "document": no interleaving


async def test_document_search_never_returns_other_advisors_documents(
    client: AsyncClient, documents: dict[str, str]
) -> None:
    hits = await _search(client, q="council tax", type="document", advisor=ADVISOR_B)

    advisor_a_ids = {documents[k] for k in ("emily_council_tax", "john_utility_bill")}
    assert advisor_a_ids.isdisjoint(hit["document"]["id"] for hit in hits)


async def test_document_search_respects_limit(
    client: AsyncClient, documents: dict[str, str]
) -> None:
    assert len(await _search(client, q="statement", type="document", limit=2)) == 2


async def _first_client_id(http: AsyncClient) -> str:
    hits = await _search(http, q="john.doe@neviswealth.com", type="client")
    return str(hits[0]["client"]["id"])
