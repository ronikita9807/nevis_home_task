"""Text embeddings: a capability shared by documents, clients and search."""

from collections.abc import Sequence
from typing import Protocol

# Fixed by the embedding model (bge-small-en-v1.5) and the `vector(384)` columns.
# Changing the model to one of another size needs a migration and a re-index.
EMBEDDING_DIMENSION = 384


class Embedder(Protocol):
    """Turns text into vectors whose cosine similarity reflects meaning.

    Queries and documents are embedded separately: retrieval models often expect a
    query-side instruction (prefix) that documents must not get.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
