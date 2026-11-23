"""Search response items: one array, discriminated by `type`."""

from typing import Literal, Self

from pydantic import Field

from app.clients.dto import ClientDTO
from app.search.ranking import RankedClient
from app.shared.domain import BaseDTO

SCORE_DESCRIPTION = (
    "Relevance within its own block (clients or documents); higher is better. "
    "Not a probability and not comparable across blocks."
)


class ClientHit(BaseDTO):
    type: Literal["client"] = "client"
    score: float = Field(description=SCORE_DESCRIPTION)
    matched_fields: list[str] = Field(description="Client fields the query matched")
    client: ClientDTO

    @classmethod
    def from_ranked(cls, ranked: RankedClient) -> Self:
        c = ranked.candidate
        return cls(
            score=ranked.score,
            matched_fields=[str(f) for f in ranked.matched_fields],
            client=ClientDTO(
                id=c.id,
                first_name=c.first_name,
                last_name=c.last_name,
                email=c.email,
                description=c.description,
                social_links=list(c.social_links),
            ),
        )


# The spec defines the response as an array. A response grouped by type
# (`{clients: [], documents: []}`) would suit a UI better and avoid ranking different
# entities together; in a real project this is a contract discussion. Here the array is
# kept and ordered in blocks (clients, then documents), each sorted by its own score.
SearchHit = ClientHit
