"""Search use case: normalise the query, fetch candidates, rank, assemble the response."""

import uuid

from app.search.domain import SearchType, normalize_query
from app.search.dto import ClientHit, SearchHit
from app.search.ports import ClientSearchQuery
from app.search.ranking import rank_clients

# Candidates fetched before ranking: generous, so the ranking (not the SQL) decides the top.
CANDIDATE_MULTIPLIER = 5


class SearchService:
    def __init__(self, clients: ClientSearchQuery) -> None:
        self._clients = clients

    async def search(
        self, advisor_id: uuid.UUID, raw_query: str, search_type: SearchType | None, limit: int
    ) -> list[SearchHit]:
        query = normalize_query(raw_query)
        hits: list[SearchHit] = []
        if search_type in (None, SearchType.CLIENT):
            candidates = await self._clients.find_clients(
                advisor_id, query, limit * CANDIDATE_MULTIPLIER
            )
            hits.extend(ClientHit.from_ranked(r) for r in rank_clients(query, candidates)[:limit])
        return hits
