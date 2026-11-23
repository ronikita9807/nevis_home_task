"""Pure ranking rules for search results (no I/O: unit-tested directly)."""

from dataclasses import dataclass

from app.search.domain import FUZZY_THRESHOLD, ClientCandidate, ClientField

EXACT_SCORE = 1.0
PREFIX_SCORE = 0.8
SUBSTRING_SCORE = 0.6
FUZZY_WEIGHT = 0.5  # fuzzy score = weight x similarity, so always below a substring match


@dataclass(frozen=True, slots=True)
class RankedClient:
    candidate: ClientCandidate
    score: float
    matched_fields: list[ClientField]


def rank_clients(query: str, candidates: list[ClientCandidate]) -> list[RankedClient]:
    """Score and order clients for a normalised query, best first, deterministically.

    Tiers: exact email or full name (1.0) > prefix of email or a name (0.8) >
    every query word found in some field (0.6) > typo-tolerant match (0.5 x similarity).
    """
    ranked = [r for c in candidates if (r := _score_client(query, c)) is not None]
    return sorted(
        ranked,
        key=lambda r: (-r.score, r.candidate.last_name, r.candidate.first_name, r.candidate.id),
    )


def _score_client(query: str, candidate: ClientCandidate) -> RankedClient | None:
    values = {
        ClientField.EMAIL: candidate.email.lower(),
        ClientField.FIRST_NAME: candidate.first_name.lower(),
        ClientField.LAST_NAME: candidate.last_name.lower(),
        ClientField.DESCRIPTION: (candidate.description or "").lower(),
    }
    words = query.split()
    matched = [
        field
        for field, value in values.items()
        if any(word in value for word in words)
        or candidate.similarity.get(field, 0.0) > FUZZY_THRESHOLD
    ]
    if not matched:
        return None

    first, last = values[ClientField.FIRST_NAME], values[ClientField.LAST_NAME]
    full_names = (f"{first} {last}", f"{last} {first}")
    if query == values[ClientField.EMAIL] or query in full_names:
        score = EXACT_SCORE
    elif any(
        name.startswith(query) for name in (values[ClientField.EMAIL], first, last, *full_names)
    ):
        score = PREFIX_SCORE
    elif all(any(word in value for value in values.values()) for word in words):
        score = SUBSTRING_SCORE
    else:
        score = FUZZY_WEIGHT * max(candidate.similarity.values(), default=0.0)
    return RankedClient(candidate=candidate, score=round(score, 4), matched_fields=matched)
