"""Pure ranking rules for search results (no I/O: unit-tested directly)."""

import re
import uuid
from dataclasses import dataclass

from app.search.domain import (
    CLIENT_SEMANTIC_BAND,
    CLIENT_SEMANTIC_FLOOR,
    CLIENT_SEMANTIC_TOP,
    CLIENT_SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    MIN_FUZZY_TOKEN_LENGTH,
    RRF_K,
    SEMANTIC_WEIGHT,
    SNIPPET_MAX_WORDS,
    ClientCandidate,
    ClientField,
    KeywordMatch,
    MatchChannel,
    SemanticClientMatch,
    SemanticMatch,
)

EXACT_SCORE = 1.0
PREFIX_SCORE = 0.8
SUBSTRING_SCORE = 0.6
FUZZY_WEIGHT = 0.5  # fuzzy score = weight x similarity, so always below a substring match

_WORD = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class RankedClient:
    candidate: ClientCandidate
    score: float
    matched_fields: list[ClientField]
    channels: tuple[MatchChannel, ...] = (MatchChannel.KEYWORD,)


def rank_clients(query: str, candidates: list[ClientCandidate]) -> list[RankedClient]:
    """Score and order clients for a normalised query, best first, deterministically.

    Every query word must be found in some field, either as a substring or as a typo of a
    word (confirmed by edit distance: the SQL trigram filter is only a cheap pre-filter).
    Tiers: exact email or full name (1.0) > prefix of email or a name (0.8) >
    every word found as a substring (0.6) > typo-tolerant match (0.5 x similarity).
    """
    ranked = [r for c in candidates if (r := _score_client(query, c)) is not None]
    return sorted(
        ranked,
        key=lambda r: (-r.score, r.candidate.last_name, r.candidate.first_name, r.candidate.id),
    )


def merge_semantic_clients(
    ranked: list[RankedClient], semantic: list[SemanticClientMatch]
) -> list[RankedClient]:
    """Add clients found by the meaning of their description to the word matches.

    Only clients above CLIENT_SEMANTIC_FLOOR and within CLIENT_SEMANTIC_BAND of the best one
    count (at most CLIENT_SEMANTIC_TOP). A client found both ways keeps its word-match score;
    a meaning-only match scores CLIENT_SEMANTIC_WEIGHT x similarity, below any word match.
    """
    accepted = [m for m in semantic if m.similarity >= CLIENT_SEMANTIC_FLOOR]
    if accepted:
        best = max(m.similarity for m in accepted)
        accepted = [m for m in accepted if m.similarity >= best - CLIENT_SEMANTIC_BAND]
    accepted = sorted(accepted, key=lambda m: -m.similarity)[:CLIENT_SEMANTIC_TOP]

    by_id = {r.candidate.id: r for r in ranked}
    for match in accepted:
        existing = by_id.get(match.candidate.id)
        if existing is not None:
            by_id[match.candidate.id] = RankedClient(
                candidate=existing.candidate,
                score=existing.score,
                matched_fields=existing.matched_fields,
                channels=(MatchChannel.KEYWORD, MatchChannel.SEMANTIC),
            )
        else:
            by_id[match.candidate.id] = RankedClient(
                candidate=match.candidate,
                score=round(CLIENT_SEMANTIC_WEIGHT * match.similarity, 4),
                matched_fields=[ClientField.DESCRIPTION],
                channels=(MatchChannel.SEMANTIC,),
            )
    return sorted(
        by_id.values(),
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
    hits = {word: _fields_matching(word, values) for word in words}
    if not all(hits.values()):
        return None
    matched = [field for field in values if any(field in fields for fields in hits.values())]

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
        score = FUZZY_WEIGHT * max(candidate.similarity.get(field, 0.0) for field in matched)
    return RankedClient(candidate=candidate, score=round(score, 4), matched_fields=matched)


def _fields_matching(word: str, values: dict[ClientField, str]) -> set[ClientField]:
    return {
        field
        for field, value in values.items()
        if word in value or (len(word) >= MIN_FUZZY_TOKEN_LENGTH and is_typo_of(word, value))
    }


def is_typo_of(word: str, text: str) -> bool:
    """`word` is within a few edits of some word of `text` ("jonh" ~ "john", "mueller" ~ "müller").

    Allowed edits: 1 for words of up to 5 letters, 2 for longer ones. A fragment of a longer
    word is not a typo: "neviswealth" does not match "wealth" (5 edits apart).
    """
    max_edits = 1 if len(word) <= 5 else 2
    return any(
        abs(len(candidate) - len(word)) <= max_edits and edit_distance(word, candidate) <= max_edits
        for candidate in _WORD.findall(text)
    )


def edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance: insertions, deletions, substitutions, transpositions."""
    previous_previous: list[int] = []
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and char_a == b[j - 2] and a[i - 2] == char_b:
                current[j] = min(current[j], previous_previous[j - 2] + 1)
        previous_previous, previous = previous, current
    return previous[-1]


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
