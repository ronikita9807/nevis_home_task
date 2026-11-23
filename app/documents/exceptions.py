"""Errors of the documents module."""

from app.shared.exceptions import ExternalServiceError, NotFoundError


class DocumentNotFoundError(NotFoundError):
    code = "document_not_found"

    def __init__(self) -> None:
        # Same message for "missing" and "owned by another advisor": existence is not leaked.
        super().__init__("Document not found")


class SummarizationError(ExternalServiceError):
    """The summarizer could not produce a summary (timeout, API error, refusal, empty reply)."""

    code = "summarization_failed"
