"""Search use case: normalise the query, fetch candidates, rank, assemble the response."""

import uuid

from app.search.domain import (
    CLIENT_SEMANTIC_TOP,
    KEYWORD_CANDIDATES,
    SEMANTIC_TOP_DOCUMENTS,
    SearchType,
    normalize_query,
)
from app.search.dto import ClientHit, DocumentHit, SearchHit
from app.search.ports import ClientSearchQuery, DocumentSearchQuery
from app.search.ranking import fuse_document_rankings, merge_semantic_clients, rank_clients
from app.search.thesaurus import expand_query
from app.shared.embeddings import Embedder

# Client candidates fetched before ranking: generous, so the ranking (not SQL) decides the top.
CANDIDATE_MULTIPLIER = 5


class SearchService:
    def __init__(
        self, clients: ClientSearchQuery, documents: DocumentSearchQuery, embedder: Embedder
    ) -> None:
        self._clients = clients
        self._documents = documents
        self._embedder = embedder

    async def search(
        self, advisor_id: uuid.UUID, raw_query: str, search_type: SearchType | None, limit: int
    ) -> list[SearchHit]:
        """Clients block first, then documents block; each ordered by its own score."""
        query = normalize_query(raw_query)
        # One inference per request, shared by both blocks (and cached across requests).
        query_vector = await self._embedder.embed_query(query)
        hits: list[SearchHit] = []
        if search_type in (None, SearchType.CLIENT):
            hits.extend(await self._search_clients(advisor_id, query, query_vector, limit))
        if search_type in (None, SearchType.DOCUMENT):
            hits.extend(await self._search_documents(advisor_id, query, query_vector, limit))
        return hits

    async def _search_clients(
        self, advisor_id: uuid.UUID, query: str, query_vector: list[float], limit: int
    ) -> list[ClientHit]:
        candidates = await self._clients.find_clients(
            advisor_id, query, limit * CANDIDATE_MULTIPLIER
        )
        semantic = await self._clients.semantic_clients(
            advisor_id, query_vector, CLIENT_SEMANTIC_TOP * CANDIDATE_MULTIPLIER
        )
        ranked = merge_semantic_clients(rank_clients(query, candidates), semantic)
        return [ClientHit.from_ranked(r) for r in ranked[:limit]]

    async def _search_documents(
        self, advisor_id: uuid.UUID, query: str, query_vector: list[float], limit: int
    ) -> list[DocumentHit]:
        semantic = await self._documents.semantic_matches(
            advisor_id, query_vector, SEMANTIC_TOP_DOCUMENTS
        )
        keyword = await self._documents.keyword_matches(
            advisor_id, query, expand_query(query), KEYWORD_CANDIDATES
        )
        fused = fuse_document_rankings(semantic, keyword)[:limit]
        details = await self._documents.document_details(
            advisor_id, [doc.document_id for doc in fused]
        )
        return [DocumentHit.from_fused(doc, details[doc.document_id]) for doc in fused]
