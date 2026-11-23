"""Search inputs and the candidate records returned by the storage layer."""

import uuid
from dataclasses import dataclass
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
