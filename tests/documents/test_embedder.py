"""The real embedding model: loads, matches the schema and fits chunks in its window."""

import json
import math
from pathlib import Path

import pytest

from app.documents.chunking import split_into_chunks
from app.documents.domain import EMBEDDING_DIMENSION
from app.documents.infra.embedder import FastEmbedEmbedder
from app.shared.settings import Settings

pytestmark = pytest.mark.semantic

MODEL_MAX_TOKENS = 512
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    defaults = Settings.model_fields
    return FastEmbedEmbedder(
        model_name=str(defaults["embedding_model"].default), cache_dir=None, threads=None
    )


def test_model_dimension_matches_database_column(embedder: FastEmbedEmbedder) -> None:
    assert embedder.dimension == EMBEDDING_DIMENSION


async def test_embeddings_are_unit_vectors(embedder: FastEmbedEmbedder) -> None:
    [vector] = await embedder.embed_documents(["Council tax bill 2024/25"])

    assert len(vector) == EMBEDDING_DIMENSION
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-5)


async def test_query_embedding_differs_from_document_embedding(
    embedder: FastEmbedEmbedder,
) -> None:
    [as_document] = await embedder.embed_documents(["proof of address"])
    as_query = await embedder.embed_query("proof of address")

    assert as_query != as_document  # the bge query instruction is applied


def test_every_demo_chunk_fits_the_model_window(embedder: FastEmbedEmbedder) -> None:
    corpus = json.loads((DATA_DIR / "demo_corpus.json").read_text())
    embedded_texts = [
        f"{doc['title']}\n{chunk}"
        for client in corpus["clients"]
        for doc in client["documents"]
        for chunk in split_into_chunks(doc["content"])
    ]

    longest = max(embedder.count_tokens(text) for text in embedded_texts)

    assert longest <= MODEL_MAX_TOKENS
