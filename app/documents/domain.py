"""Document entity, its searchable chunks and the field types shared with commands."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints

from app.shared.domain import BaseDomainModel

# Fixed by the embedding model (bge-small-en-v1.5) and the `vector(384)` column.
# Changing the model to one of another size needs a migration and a re-index.
EMBEDDING_DIMENSION = 384

MAX_TITLE_LENGTH = 300
# Characters. Indexing is synchronous (~20 s at this size on a laptop CPU); larger documents
# would need asynchronous indexing.
MAX_CONTENT_LENGTH = 100_000

Title = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TITLE_LENGTH)
]
Content = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CONTENT_LENGTH)
]
Embedding = Annotated[
    list[float], Field(min_length=EMBEDDING_DIMENSION, max_length=EMBEDDING_DIMENSION)
]


class SummaryMethod(StrEnum):
    LLM = "llm"
    EXTRACTIVE = "extractive"


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    text: str
    method: SummaryMethod


class Document(BaseDomainModel):
    """A client's document. Its owner is always the owner of the client."""

    id: uuid.UUID
    client_id: uuid.UUID
    advisor_id: uuid.UUID
    title: Title
    content: Content
    created_at: datetime
    # Computed on first request and cached: documents are immutable, so it never goes stale.
    summary: DocumentSummary | None = None

    @classmethod
    def create(
        cls, *, client_id: uuid.UUID, advisor_id: uuid.UUID, title: str, content: str
    ) -> Self:
        return cls(
            id=uuid.uuid4(),
            client_id=client_id,
            advisor_id=advisor_id,
            title=title,
            content=content,
            created_at=datetime.now(UTC),
        )


class DocumentChunk(BaseDomainModel):
    """A searchable slice of a document with its embedding."""

    index: Annotated[int, Field(ge=0)]
    text: Annotated[str, StringConstraints(min_length=1)]
    embedding: Embedding
