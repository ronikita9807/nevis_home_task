"""Load the demo corpus (data/demo_corpus.json) into the API under test."""

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import status
from httpx import AsyncClient

from tests.helpers import as_advisor

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_json(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DATA_DIR / name).read_text())
    return data


async def seed_demo_corpus(http: AsyncClient) -> dict[str, str]:
    """Create every advisor's clients and documents; return document key -> document id."""
    corpus = load_json("demo_corpus.json")
    advisors = {a["key"]: uuid.UUID(a["id"]) for a in corpus["advisors"]}
    document_ids: dict[str, str] = {}
    for client in corpus["clients"]:
        headers = as_advisor(advisors[client["advisor"]])
        body = {
            k: client[k]
            for k in ("first_name", "last_name", "email", "description", "social_links")
        }
        created = await http.post("/clients", json=body, headers=headers)
        assert created.status_code == status.HTTP_201_CREATED, created.text
        for document in client["documents"]:
            response = await http.post(
                f"/clients/{created.json()['id']}/documents",
                json={"title": document["title"], "content": document["content"]},
                headers=headers,
            )
            assert response.status_code == status.HTTP_201_CREATED, response.text
            document_ids[document["key"]] = response.json()["id"]
    return document_ids
