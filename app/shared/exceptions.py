"""Domain exception hierarchy. Mapped to HTTP responses in one place (`problems.py`)."""

from typing import Any, ClassVar


class AppError(Exception):
    """Base for expected application errors.

    `code` is a stable machine-readable identifier exposed to API clients.
    """

    code: ClassVar[str] = "app_error"

    def __init__(self, detail: str, *, details: dict[str, Any] | None = None) -> None:
        # Any: details are free-form, JSON-serialisable context for the client.
        super().__init__(detail)
        self.detail = detail
        self.details = details or {}


class NotFoundError(AppError):
    code = "not_found"


class ConflictError(AppError):
    code = "conflict"


class UnauthorizedError(AppError):
    code = "unauthorized"


class ServiceUnavailableError(AppError):
    code = "service_unavailable"


class ExternalServiceError(AppError):
    code = "external_service_error"
