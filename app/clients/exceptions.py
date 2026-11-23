"""Errors of the clients module."""

from app.shared.exceptions import ConflictError, NotFoundError


class ClientNotFoundError(NotFoundError):
    code = "client_not_found"

    def __init__(self) -> None:
        # Same message for "missing" and "owned by another advisor": existence is not leaked.
        super().__init__("Client not found")


class DuplicateClientEmailError(ConflictError):
    code = "duplicate_client_email"

    def __init__(self) -> None:
        super().__init__("You already have a client with this email")
