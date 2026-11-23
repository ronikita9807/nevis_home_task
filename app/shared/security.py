"""Caller authentication dependencies.

Trust model (MVP): the service sits behind a trusted gateway/BFF that authenticates the
advisor and forwards their id in `X-Advisor-Id`. The shared service key `X-API-Key`
proves the call comes from that trusted caller; it is checked only when configured.
"""

import secrets
import uuid
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.containers import ApplicationContainer
from app.shared.exceptions import UnauthorizedError

_service_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ServiceKey",
    description="Shared key of the trusted caller. Required only when the server sets API_KEY.",
    auto_error=False,
)
_advisor_header = APIKeyHeader(
    name="X-Advisor-Id",
    scheme_name="AdvisorId",
    description="UUID of the advisor on whose behalf the call is made. All data is scoped to it.",
    auto_error=False,
)


@inject
def require_service_key(
    provided: Annotated[str | None, Security(_service_key_header)],
    expected: str | None = Depends(Provide[ApplicationContainer.config.api_key]),
) -> None:
    if expected is None:
        return
    if provided is None or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise UnauthorizedError("Missing or invalid X-API-Key")


def current_advisor_id(raw: Annotated[str | None, Security(_advisor_header)]) -> uuid.UUID:
    if not raw:
        raise UnauthorizedError("Missing X-Advisor-Id header")
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        raise UnauthorizedError("X-Advisor-Id must be a UUID") from exc


# A plain alias (not a `type` statement): FastAPI must see the Annotated metadata.
AdvisorId = Annotated[uuid.UUID, Depends(current_advisor_id)]
