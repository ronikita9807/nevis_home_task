"""Liveness and readiness probes."""

import logging

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from app.containers import ApplicationContainer
from app.shared.db import Database
from app.shared.exceptions import ServiceUnavailableError
from app.shared.problems import problem_responses

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])


class StatusDTO(BaseModel):
    status: str


@router.get("/health", summary="Liveness probe")
def health() -> StatusDTO:
    """Return 200 while the process is alive. Dependencies are not checked."""
    return StatusDTO(status="ok")


@router.get(
    "/ready",
    summary="Readiness probe",
    responses=problem_responses(status.HTTP_503_SERVICE_UNAVAILABLE),
)
@inject
async def ready(
    database: Database = Depends(Provide[ApplicationContainer.database]),  # noqa: B008
) -> StatusDTO:
    """Return 200 when the service can serve traffic (database reachable), otherwise 503."""
    try:
        await database.ping()
    except Exception as exc:  # any connectivity failure means "not ready"
        logger.warning("Readiness check failed: %s", type(exc).__name__)
        raise ServiceUnavailableError("Database is unavailable") from exc
    return StatusDTO(status="ready")
