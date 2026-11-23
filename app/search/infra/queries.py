"""PostgreSQL read queries for search.

This module reads the clients and documents tables directly: a read model that spans
modules is the allowed exception to "modules talk through their public ports".
"""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, String, and_, cast, func, literal, or_, select
from sqlalchemy.dialects.postgresql import REGCONFIG

from app.clients.infra.orm import ClientRow
from app.documents.infra.orm import FTS_CONFIG, DocumentChunkRow, DocumentRow
from app.search.domain import (
    FUZZY_THRESHOLD,
    MIN_FUZZY_TOKEN_LENGTH,
    SEMANTIC_CHUNK_CANDIDATES,
    SEMANTIC_NOISE_FLOOR,
    ClientCandidate,
    ClientField,
    DocumentDetails,
    KeywordMatch,
    SemanticMatch,
)
from app.shared.db import Database

LIKE_ESCAPE = "\\"

# Bare columns in WHERE so the trigram indexes apply (a NULL description just fails to match).
_CLIENT_COLUMNS: dict[ClientField, ColumnElement[str] | ColumnElement[str | None]] = {
    ClientField.EMAIL: ClientRow.email.expression,
    ClientField.FIRST_NAME: ClientRow.first_name.expression,
    ClientField.LAST_NAME: ClientRow.last_name.expression,
    ClientField.DESCRIPTION: ClientRow.description.expression,
}


def escape_like(value: str) -> str:
    """Make LIKE wildcards literal: a query of `%` or `_` must not match everything."""
    return (
        value.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2)
        .replace("%", f"{LIKE_ESCAPE}%")
        .replace("_", f"{LIKE_ESCAPE}_")
    )


def _word_matches(word: str) -> ColumnElement[bool]:
    pattern = f"%{escape_like(word)}%"
    conditions = [column.ilike(pattern, escape=LIKE_ESCAPE) for column in _CLIENT_COLUMNS.values()]
    if len(word) >= MIN_FUZZY_TOKEN_LENGTH:
        # `word <% column`: the column contains a word similar to `word` (pg_trgm, GIN index).
        conditions += [
            literal(word, String).op("<%")(column) for column in _CLIENT_COLUMNS.values()
        ]
    return or_(*conditions)


class PostgreSQLClientSearchQuery:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def find_clients(
        self, advisor_id: uuid.UUID, query: str, limit: int
    ) -> list[ClientCandidate]:
        similarities = {
            field: func.word_similarity(query, func.coalesce(column, "")).label(f"sim_{field}")
            for field, column in _CLIENT_COLUMNS.items()
        }
        statement = (
            select(ClientRow, *similarities.values())
            .where(ClientRow.visible_to(advisor_id), *(_word_matches(w) for w in query.split()))
            .order_by(func.greatest(*similarities.values()).desc(), ClientRow.id)
            .limit(limit)
        )
        async with self._database.session() as session, session.begin():
            # Scoped to this transaction: the `<%` operator uses this threshold.
            await session.execute(
                select(
                    func.set_config("pg_trgm.word_similarity_threshold", str(FUZZY_THRESHOLD), True)
                )
            )
            rows = (await session.execute(statement)).all()

        candidates: list[ClientCandidate] = []
        for client, *scores in rows:
            candidates.append(
                ClientCandidate(
                    id=client.id,
                    first_name=client.first_name,
                    last_name=client.last_name,
                    email=client.email,
                    description=client.description,
                    social_links=list(client.social_links),
                    similarity={
                        field: float(score)
                        for field, score in zip(similarities, scores, strict=True)
                    },
                )
            )
        return candidates


# HNSW explores this many candidates (default 40 would silently cap the chunk scan).
HNSW_EF_SEARCH = 100
HEADLINE_OPTIONS = 'MaxWords=35, MinWords=15, MaxFragments=1, StartSel="", StopSel=""'


class PostgreSQLDocumentSearchQuery:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def semantic_matches(
        self, advisor_id: uuid.UUID, query_vector: Sequence[float], top_documents: int
    ) -> list[SemanticMatch]:
        distance = DocumentChunkRow.embedding.cosine_distance(list(query_vector))
        statement = (
            select(DocumentChunkRow.document_id, DocumentChunkRow.text, distance.label("distance"))
            .where(DocumentChunkRow.visible_to(advisor_id))
            .order_by(distance)
            .limit(SEMANTIC_CHUNK_CANDIDATES)
        )
        async with self._database.session() as session, session.begin():
            await session.execute(
                select(
                    func.set_config("hnsw.ef_search", str(HNSW_EF_SEARCH), True),
                    # Keep scanning the index until enough rows pass the advisor filter,
                    # instead of filtering a fixed candidate set down to too few results.
                    func.set_config("hnsw.iterative_scan", "relaxed_order", True),
                )
            )
            rows = (await session.execute(statement)).all()

        best: dict[uuid.UUID, SemanticMatch] = {}
        # relaxed_order may return rows slightly out of order: sort before picking the best.
        for document_id, chunk_text, chunk_distance in sorted(rows, key=lambda r: r.distance):
            if 1 - float(chunk_distance) < SEMANTIC_NOISE_FLOOR:
                break
            if document_id not in best:
                best[document_id] = SemanticMatch(
                    document_id=document_id,
                    similarity=1 - float(chunk_distance),
                    chunk_text=chunk_text,
                )
        return list(best.values())[:top_documents]

    async def keyword_matches(
        self, advisor_id: uuid.UUID, query: str, expansions: Sequence[str], limit: int
    ) -> list[KeywordMatch]:
        config = cast(literal(FTS_CONFIG), REGCONFIG)
        # Any: SQLAlchemy has no static type for tsquery expressions.
        tsquery: ColumnElement[Any] = func.websearch_to_tsquery(config, query)
        for phrase in expansions:
            tsquery = tsquery.op("||")(func.phraseto_tsquery(config, phrase))
        rank = func.ts_rank_cd(DocumentRow.search_tsv, tsquery)
        headline = func.ts_headline(config, DocumentRow.content, tsquery, HEADLINE_OPTIONS)
        statement = (
            select(DocumentRow.id, rank.label("rank"), headline.label("headline"))
            .where(DocumentRow.visible_to(advisor_id), DocumentRow.search_tsv.op("@@")(tsquery))
            .order_by(rank.desc(), DocumentRow.id)
            .limit(limit)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return [
            KeywordMatch(document_id=row.id, rank=float(row.rank), headline=row.headline)
            for row in rows
        ]

    async def document_details(
        self, advisor_id: uuid.UUID, document_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, DocumentDetails]:
        if not document_ids:
            return {}
        statement = (
            select(
                DocumentRow.id,
                DocumentRow.client_id,
                DocumentRow.title,
                DocumentRow.created_at,
                ClientRow.first_name,
                ClientRow.last_name,
            )
            .join(
                ClientRow,
                and_(
                    ClientRow.id == DocumentRow.client_id,
                    ClientRow.advisor_id == DocumentRow.advisor_id,
                ),
            )
            .where(DocumentRow.visible_to(advisor_id), DocumentRow.id.in_(document_ids))
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return {
            row.id: DocumentDetails(
                id=row.id,
                client_id=row.client_id,
                title=row.title,
                created_at=row.created_at,
                client_first_name=row.first_name,
                client_last_name=row.last_name,
            )
            for row in rows
        }
