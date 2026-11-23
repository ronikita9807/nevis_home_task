"""PostgreSQL read queries for search.

This module reads the clients (and later documents) tables directly: a read model that
spans modules is the allowed exception to "modules talk through their public ports".
"""

import uuid

from sqlalchemy import ColumnElement, String, func, literal, or_, select

from app.clients.infra.orm import ClientRow
from app.search.domain import (
    FUZZY_THRESHOLD,
    MIN_FUZZY_TOKEN_LENGTH,
    ClientCandidate,
    ClientField,
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
