"""End-to-end search quality on the golden set, with the real embedding model.

Runs every query of data/golden_queries.json through GET /search (documents only) after
seeding data/demo_corpus.json, and checks recall@5 and MRR. `make eval` prints the table.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from dependency_injector import providers
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.shared.fastembed_embedder import FastEmbedEmbedder
from app.shared.settings import Settings
from tests.helpers import as_advisor
from tests.search.corpus import load_json, seed_demo_corpus

pytestmark = pytest.mark.semantic

TOP_K = 5
# Floors a little under the measured values (see `make eval`), leaving room for tiny
# floating-point differences between CPUs; a drop below them means search got worse.
MIN_RECALL_AT_5 = 0.85
MIN_MRR = 0.8


@pytest.fixture(scope="module")
def real_embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder(
        model_name=str(Settings.model_fields["embedding_model"].default),
        cache_dir=None,
        threads=None,
    )


@pytest.fixture
async def real_client(
    settings: Settings, real_embedder: FastEmbedEmbedder
) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    app.state.container.embedder.override(providers.Object(real_embedder))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
    await app.state.container.database().dispose()


async def _ranked_document_ids(http: AsyncClient, advisor: uuid.UUID, query: str) -> list[str]:
    response = await http.get(
        "/search",
        params={"q": query, "type": "document", "limit": TOP_K},
        headers=as_advisor(advisor),
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    return [hit["document"]["id"] for hit in response.json()]


async def test_golden_set_recall_and_mrr(real_client: AsyncClient) -> None:
    document_ids = await seed_demo_corpus(real_client)
    advisors = {a["key"]: uuid.UUID(a["id"]) for a in load_json("demo_corpus.json")["advisors"]}
    queries: list[dict[str, Any]] = load_json("golden_queries.json")["queries"]

    rows: list[tuple[str, float, float]] = []
    for query in queries:
        relevant = {document_ids[key] for key in query["relevant"]}
        ranking = await _ranked_document_ids(
            real_client, advisors[query["advisor"]], query["query"]
        )
        recall = len(relevant & set(ranking)) / len(relevant)
        first = next((i for i, doc in enumerate(ranking, start=1) if doc in relevant), None)
        rows.append((query["query"], recall, 1 / first if first else 0.0))

    recall_at_5 = sum(r for _, r, _ in rows) / len(rows)
    mrr = sum(m for _, _, m in rows) / len(rows)
    print(f"\n| query | recall@{TOP_K} | reciprocal rank |\n|---|---|---|")
    for text, recall, rr in rows:
        print(f"| {text} | {recall:.2f} | {rr:.2f} |")
    print(f"| **mean** | **{recall_at_5:.2f}** | **{mrr:.2f}** |")

    assert recall_at_5 >= MIN_RECALL_AT_5
    assert mrr >= MIN_MRR


@pytest.mark.parametrize("query", ["NevisWealth", "Jonh", "Raj Patel"])
async def test_name_like_queries_do_not_drag_in_unrelated_documents(
    real_client: AsyncClient, query: str
) -> None:
    await seed_demo_corpus(real_client)
    advisor = uuid.UUID(load_json("demo_corpus.json")["advisors"][0]["id"])

    response = await real_client.get(
        "/search", params={"q": query, "type": "document"}, headers=as_advisor(advisor)
    )

    assert [h for h in response.json() if h["matched_by"] == ["semantic"]] == []


async def test_address_proof_ranks_proof_of_address_documents_on_top(
    real_client: AsyncClient,
) -> None:
    document_ids = await seed_demo_corpus(real_client)
    advisor = uuid.UUID(load_json("demo_corpus.json")["advisors"][0]["id"])

    top3 = await _ranked_document_ids(real_client, advisor, "address proof")

    expected = {document_ids[k] for k in ("john_utility_bill", "emily_council_tax")}
    assert expected <= set(top3[:3])


@pytest.mark.parametrize(
    ("query", "expected_last_name"),
    [
        ("lawyer", "O'Brien"),
        ("doctor", "Patel"),
        ("risk-averse investor", "Carter"),
        ("moving to Britain", "García"),
    ],
)
async def test_clients_found_by_meaning_of_description(
    real_client: AsyncClient, query: str, expected_last_name: str
) -> None:
    await seed_demo_corpus(real_client)
    advisor = uuid.UUID(load_json("demo_corpus.json")["advisors"][0]["id"])

    response = await real_client.get(
        "/search", params={"q": query, "type": "client"}, headers=as_advisor(advisor)
    )

    [first, *_] = response.json()
    assert first["client"]["last_name"] == expected_last_name
    assert first["matched_by"] == ["semantic"]


@pytest.mark.parametrize("query", ["NevisWealth", "Jonh", "Raj Patel", "Carter"])
async def test_name_like_queries_find_no_clients_by_meaning(
    real_client: AsyncClient, query: str
) -> None:
    await seed_demo_corpus(real_client)
    advisor = uuid.UUID(load_json("demo_corpus.json")["advisors"][0]["id"])

    response = await real_client.get(
        "/search", params={"q": query, "type": "client"}, headers=as_advisor(advisor)
    )

    assert [h for h in response.json() if h["matched_by"] == ["semantic"]] == []
