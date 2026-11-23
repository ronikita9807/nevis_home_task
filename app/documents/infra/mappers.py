"""Explicit domain <-> storage mapping for documents."""

from app.documents.domain import Document, DocumentChunk, DocumentSummary, SummaryMethod
from app.documents.infra.orm import DocumentChunkRow, DocumentRow


def document_to_row(document: Document) -> DocumentRow:
    return DocumentRow(
        id=document.id,
        client_id=document.client_id,
        advisor_id=document.advisor_id,
        title=document.title,
        content=document.content,
        created_at=document.created_at,
    )


def document_to_domain(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        client_id=row.client_id,
        advisor_id=row.advisor_id,
        title=row.title,
        content=row.content,
        created_at=row.created_at,
        summary=summary_to_domain(row.summary, row.summary_method),
    )


def summary_to_domain(text: str | None, method: str | None) -> DocumentSummary | None:
    # The database guarantees both are set or both are NULL.
    if text is None or method is None:
        return None
    return DocumentSummary(text=text, method=SummaryMethod(method))


def chunk_to_row(
    chunk: DocumentChunk, document: Document, embedding_model: str
) -> DocumentChunkRow:
    return DocumentChunkRow(
        document_id=document.id,
        advisor_id=document.advisor_id,
        chunk_index=chunk.index,
        text=chunk.text,
        embedding=chunk.embedding,
        embedding_model=embedding_model,
    )
