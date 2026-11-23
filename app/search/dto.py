"""Search response items: one array, discriminated by `type`."""

import uuid
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field

from app.clients.dto import ClientDTO
from app.search.domain import DocumentDetails
from app.search.ranking import FusedDocument, RankedClient
from app.shared.domain import BaseDTO

SCORE_DESCRIPTION = (
    "Relevance within its own block (clients or documents); higher is better. "
    "Not a probability and not comparable across blocks."
)


class ClientHit(BaseDTO):
    type: Literal["client"] = "client"
    score: float = Field(description=SCORE_DESCRIPTION)
    matched_fields: list[str] = Field(description="Client fields the query matched")
    matched_by: list[str] = Field(
        description="`keyword` (words, typos) and/or `semantic` (meaning of the description)"
    )
    client: ClientDTO

    @classmethod
    def from_ranked(cls, ranked: RankedClient) -> Self:
        c = ranked.candidate
        return cls(
            score=ranked.score,
            matched_fields=[str(f) for f in ranked.matched_fields],
            matched_by=[str(channel) for channel in ranked.channels],
            client=ClientDTO(
                id=c.id,
                first_name=c.first_name,
                last_name=c.last_name,
                email=c.email,
                description=c.description,
                social_links=list(c.social_links),
            ),
        )


class DocumentSummaryDTO(BaseDTO):
    """The spec `Document` without its full content (the hit carries a snippet instead)."""

    id: uuid.UUID
    client_id: uuid.UUID
    title: str
    created_at: datetime


class ClientRefDTO(BaseDTO):
    """Whose document it is, so the advisor does not need a second request."""

    id: uuid.UUID
    first_name: str
    last_name: str


class DocumentHit(BaseDTO):
    type: Literal["document"] = "document"
    score: float = Field(description=SCORE_DESCRIPTION)
    matched_by: list[str] = Field(
        description="`semantic` (similar meaning) and/or `keyword` (words or known synonyms)"
    )
    snippet: str = Field(description="Most relevant passage of the document")
    document: DocumentSummaryDTO
    client: ClientRefDTO

    @classmethod
    def from_fused(cls, fused: FusedDocument, details: DocumentDetails) -> Self:
        return cls(
            score=fused.score,
            matched_by=[str(channel) for channel in fused.channels],
            snippet=fused.snippet,
            document=DocumentSummaryDTO(
                id=details.id,
                client_id=details.client_id,
                title=details.title,
                created_at=details.created_at,
            ),
            client=ClientRefDTO(
                id=details.client_id,
                first_name=details.client_first_name,
                last_name=details.client_last_name,
            ),
        )


# The spec defines the response as an array. A response grouped by type
# (`{clients: [], documents: []}`) would suit a UI better and avoid ranking different
# entities together; in a real project this is a contract discussion. Here the array is
# kept and ordered in blocks (clients, then documents), each sorted by its own score.
SearchHit = Annotated[ClientHit | DocumentHit, Field(discriminator="type")]
