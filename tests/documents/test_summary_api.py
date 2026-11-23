from typing import Any

import pytest
from dependency_injector import providers
from fastapi import FastAPI, status
from httpx import AsyncClient

from app.documents.chunking import split_sentences
from app.documents.domain import SummaryMethod
from app.documents.exceptions import SummarizationError
from tests.helpers import ADVISOR_A, ADVISOR_B, as_advisor

CONTENT = " ".join(
    f"Statement line {i}: contributions of £{i}00 were paid into the pension." for i in range(8)
)


class FakeLLMSummarizer:
    method = SummaryMethod.LLM

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def summarize(self, title: str, content: str) -> str:
        self.calls += 1
        if self.fail:
            raise SummarizationError("model unavailable")
        return f"Summary of {title}."


@pytest.fixture
async def document_url(client: AsyncClient) -> str:
    created = await client.post(
        "/clients",
        json={"first_name": "John", "last_name": "Doe", "email": "john@example.com"},
        headers=as_advisor(ADVISOR_A),
    )
    document = await client.post(
        f"/clients/{created.json()['id']}/documents",
        json={"title": "Pension statement", "content": CONTENT},
        headers=as_advisor(ADVISOR_A),
    )
    return str(document.headers["Location"])


def _use_llm(app: FastAPI, llm: FakeLLMSummarizer) -> None:
    app.state.container.llm_summarizer.override(providers.Object(llm))


async def _summary(client: AsyncClient, url: str) -> dict[str, Any]:
    response = await client.get(f"{url}/summary", headers=as_advisor(ADVISOR_A))
    assert response.status_code == status.HTTP_200_OK, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_summary_without_llm_is_extractive(client: AsyncClient, document_url: str) -> None:
    body = await _summary(client, document_url)

    sentences = split_sentences(body["summary"])
    assert body["method"] == "extractive"
    assert len(sentences) == 3
    assert all(sentence in split_sentences(CONTENT) for sentence in sentences)


async def test_summary_with_llm_is_cached(
    app: FastAPI, client: AsyncClient, document_url: str
) -> None:
    llm = FakeLLMSummarizer()
    _use_llm(app, llm)

    first = await _summary(client, document_url)
    second = await _summary(client, document_url)

    assert first == second
    assert first["method"] == "llm"
    assert first["summary"] == "Summary of Pension statement."
    assert llm.calls == 1


async def test_summary_falls_back_to_extractive_when_llm_fails(
    app: FastAPI, client: AsyncClient, document_url: str
) -> None:
    _use_llm(app, FakeLLMSummarizer(fail=True))

    body = await _summary(client, document_url)

    assert body["method"] == "extractive"


async def test_summary_of_other_advisors_document_returns_404(
    client: AsyncClient, document_url: str
) -> None:
    response = await client.get(f"{document_url}/summary", headers=as_advisor(ADVISOR_B))

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["code"] == "document_not_found"
