"""Pure ranking rules for search results (no I/O: unit-tested directly)."""

import uuid
from dataclasses import dataclass

from app.search.domain import (
    FUZZY_THRESHOLD,
    KEYWORD_WEIGHT,
    RRF_K,
    SEMANTIC_WEIGHT,
    SNIPPET_MAX_WORDS,
    ClientCandidate,
    ClientField,
    KeywordMatch,
    MatchChannel,
    SemanticMatch,
)

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


@dataclass(frozen=True, slots=True)
class FusedDocument:
    document_id: uuid.UUID
    score: float
    channels: list[MatchChannel]
    snippet: str


def fuse_document_rankings(
    semantic: list[SemanticMatch], keyword: list[KeywordMatch], k: int = RRF_K
) -> list[FusedDocument]:
    """Weighted Reciprocal Rank Fusion: score(d) = sum over channels of w / (k + rank of d).

    Ranks, not raw scores, are combined: cosine similarity and ts_rank are not comparable.
    A document found by both channels rises above one found by either alone; keyword
    matches weigh more than semantic ones (see KEYWORD_WEIGHT).
    The snippet prefers the keyword headline (it shows the matched words), else the best chunk.
    """
    scores: dict[uuid.UUID, float] = {}
    channels: dict[uuid.UUID, list[MatchChannel]] = {}
    snippets: dict[uuid.UUID, str] = {}
    for rank, keyword_match in enumerate(keyword, start=1):
        doc_id = keyword_match.document_id
        scores[doc_id] = scores.get(doc_id, 0.0) + KEYWORD_WEIGHT / (k + rank)
        channels.setdefault(doc_id, []).append(MatchChannel.KEYWORD)
        snippets[doc_id] = keyword_match.headline
    for rank, semantic_match in enumerate(semantic, start=1):
        doc_id = semantic_match.document_id
        scores[doc_id] = scores.get(doc_id, 0.0) + SEMANTIC_WEIGHT / (k + rank)
        channels.setdefault(doc_id, []).insert(0, MatchChannel.SEMANTIC)
        snippets.setdefault(doc_id, truncate_words(semantic_match.chunk_text))

    fused = [
        FusedDocument(
            document_id=doc_id,
            score=round(score, 6),
            channels=channels[doc_id],
            snippet=snippets[doc_id],
        )
        for doc_id, score in scores.items()
    ]
    return sorted(fused, key=lambda d: (-d.score, str(d.document_id)))


def truncate_words(text: str, max_words: int = SNIPPET_MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words) if len(words) <= max_words else " ".join(words[:max_words]) + " …"
