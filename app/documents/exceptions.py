"""Errors of the documents module."""

from app.shared.exceptions import NotFoundError


class DocumentNotFoundError(NotFoundError):
    code = "document_not_found"

    def __init__(self) -> None:
        # Same message for "missing" and "owned by another advisor": existence is not leaked.
        super().__init__("Document not found")
