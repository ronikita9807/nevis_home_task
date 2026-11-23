"""Search inputs and the candidate records returned by the storage layer."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import StringConstraints

MAX_QUERY_LENGTH = 256
DEFAULT_LIMIT = 20
MAX_LIMIT = 100

# A query word must be at least this long to be matched fuzzily (typos); shorter words
# produce too few trigrams and match noise.
MIN_FUZZY_TOKEN_LENGTH = 4
# pg_trgm word similarity above which a word counts as a typo of a field's word.
# Measured: "jonh"~"John" 0.40, "mueller"~"Müller" 0.375 pass; "emily"~"Emma" 0.33 does not.
FUZZY_THRESHOLD = 0.35

SearchText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)
]


class SearchType(StrEnum):
    CLIENT = "client"
    DOCUMENT = "document"


class ClientField(StrEnum):
    EMAIL = "email"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    DESCRIPTION = "description"


def normalize_query(raw: str) -> str:
    """Lower-case and collapse whitespace: `  John   DOE ` -> `john doe`."""
    return " ".join(raw.lower().split())


@dataclass(frozen=True, slots=True)
class ClientCandidate:
    """A client that matched the query in storage, with per-field fuzzy similarity."""

    id: uuid.UUID
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: list[str]
    similarity: dict[ClientField, float]


# --- Documents ----------------------------------------------------------------------

# No similarity cut-off separates relevant from irrelevant documents (their cosine scores
# overlap, docs/embedding-experiment.md), so the semantic channel proposes a fixed number of
# best documents; keyword matches are always included; RRF orders the union.
SEMANTIC_TOP_DOCUMENTS = 5
SEMANTIC_CHUNK_CANDIDATES = 100  # nearest chunks scanned to find the best documents
# A noise floor, not a relevance cut-off: below it a chunk is unrelated to the query (names,
# emails, company names). Measured on the golden set: recall/MRR are unchanged up to 0.55 and
# collapse at 0.6 (0.92 -> 0.58); at 0.5 queries such as "NevisWealth" or "Jonh" no longer
# drag in unrelated documents.
SEMANTIC_NOISE_FLOOR = 0.5
KEYWORD_CANDIDATES = 50
RRF_K = 60  # standard Reciprocal Rank Fusion constant
# Weighted RRF: a keyword match (the query's words or a curated synonym) is high-precision,
# while the semantic channel has no reliable cut-off and its top hits are noisy on domain
# queries (for "address proof" it ranks passports first). Queries sharing no words with any
# document leave the keyword channel empty, so the weight only matters where both fire.
KEYWORD_WEIGHT = 2.0
SEMANTIC_WEIGHT = 1.0
SNIPPET_MAX_WORDS = 40


class MatchChannel(StrEnum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    """A document whose best chunk is among the nearest to the query embedding."""

    document_id: uuid.UUID
    similarity: float
    chunk_text: str


@dataclass(frozen=True, slots=True)
class KeywordMatch:
    """A document matching the query (or its thesaurus expansions) in full-text search."""

    document_id: uuid.UUID
    rank: float
    headline: str


@dataclass(frozen=True, slots=True)
class DocumentDetails:
    id: uuid.UUID
    client_id: uuid.UUID
    title: str
    created_at: datetime
    client_first_name: str
    client_last_name: str
