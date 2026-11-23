"""Ports of the search module (read side only)."""

import uuid
from collections.abc import Sequence
from typing import Protocol

from app.search.domain import ClientCandidate, DocumentDetails, KeywordMatch, SemanticMatch


class ClientSearchQuery(Protocol):
    async def find_clients(
        self, advisor_id: uuid.UUID, query: str, limit: int
    ) -> list[ClientCandidate]:
        """The advisor's clients where every query word matches some field (or a typo of it).

        `query` is normalised (lower-case, single spaces). At most `limit` candidates.
        """
        ...


class DocumentSearchQuery(Protocol):
    async def semantic_matches(
        self, advisor_id: uuid.UUID, query_vector: Sequence[float], top_documents: int
    ) -> list[SemanticMatch]:
        """The advisor's documents whose chunks are nearest to the query vector, best first."""
        ...

    async def keyword_matches(
        self, advisor_id: uuid.UUID, query: str, expansions: Sequence[str], limit: int
    ) -> list[KeywordMatch]:
        """The advisor's documents matching the query or any expansion phrase, best first."""
        ...

    async def document_details(
        self, advisor_id: uuid.UUID, document_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, DocumentDetails]:
        """Title, dates and owning client of the advisor's documents, by id."""
        ...
