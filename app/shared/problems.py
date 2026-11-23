"""RFC 9457 Problem Details: the single error format of the API.

All errors (domain, validation, routing, unexpected) are rendered here, so clients
handle one shape: `{type, title, status, detail, instance, code, request_id, errors?}`.
"""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.shared.exceptions import (
    AppError,
    ConflictError,
    ExternalServiceError,
    NotFoundError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from app.shared.observability import REQUEST_ID_HEADER, request_id_var

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"

_STATUS_BY_ERROR: dict[type[AppError], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
    UnauthorizedError: status.HTTP_401_UNAUTHORIZED,
    ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ExternalServiceError: status.HTTP_502_BAD_GATEWAY,
}


class FieldError(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ProblemDetails(BaseModel):
    """Error response body (RFC 9457)."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str
    code: str
    request_id: str | None = None
    errors: list[FieldError] | None = None


def problem_response(
    request: Request,
    status_code: int,
    *,
    code: str,
    detail: str,
    errors: list[FieldError] | None = None,
) -> JSONResponse:
    request_id = request_id_var.get()
    body = ProblemDetails(
        title=HTTPStatus(status_code).phrase,
        status=status_code,
        detail=detail,
        instance=request.url.path,
        code=code,
        request_id=request_id,
        errors=errors,
    )
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


async def _handle_app_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)  # noqa: S101 - narrowing for the type checker
    status_code = next(
        (code for cls, code in _STATUS_BY_ERROR.items() if isinstance(exc, cls)),
        status.HTTP_400_BAD_REQUEST,
    )
    return problem_response(request, status_code, code=exc.code, detail=exc.detail)


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101 - narrowing for the type checker
    # Any: FastAPI types validation errors as free-form JSON-ready dicts.
    raw_errors: list[dict[str, Any]] = jsonable_encoder(exc.errors())
    errors = [
        FieldError(loc=list(error["loc"]), msg=error["msg"], type=error["type"])
        for error in raw_errors
    ]
    return problem_response(
        request,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="validation_error",
        detail="Request validation failed",
        errors=errors,
    )


async def _handle_http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - narrowing for the type checker
    status_code = exc.status_code
    code = HTTPStatus(status_code).name.lower()
    return problem_response(request, status_code, code=code, detail=str(exc.detail))


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error", exc_info=exc)
    return problem_response(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        detail="Internal server error",
    )


def register_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)


# Shared OpenAPI `responses=` entries so every route documents the same error shape.
# Any: FastAPI's `responses=` parameter is itself typed as free-form OpenAPI dicts.
def problem_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    return {
        code: {
            "model": ProblemDetails,
            "description": HTTPStatus(code).phrase,
            "content": {PROBLEM_MEDIA_TYPE: {}},
        }
        for code in status_codes
    }
