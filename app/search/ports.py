"""Ports of the search module (read side only)."""

import uuid
from typing import Protocol

from app.search.domain import ClientCandidate


class ClientSearchQuery(Protocol):
    async def find_clients(
        self, advisor_id: uuid.UUID, query: str, limit: int
    ) -> list[ClientCandidate]:
        """The advisor's clients where every query word matches some field (or a typo of it).

        `query` is normalised (lower-case, single spaces). At most `limit` candidates.
        """
        ...
