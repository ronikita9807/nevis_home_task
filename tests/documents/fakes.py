"""Deterministic stand-in for the embedding model (no weights, instant)."""

import hashlib
import math
import re
from collections.abc import Sequence

from app.documents.domain import EMBEDDING_DIMENSION

_WORD = re.compile(r"\w+")


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector: texts sharing words are close."""

    model_name = "fake-hash-embedder"
    dimension = EMBEDDING_DIMENSION

    def __init__(self) -> None:
        self.calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSION
        for word in _WORD.findall(text.lower()):
            bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4]) % EMBEDDING_DIMENSION  # noqa: S324 - not security
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
