"""HTTP endpoints of the clients module."""

import uuid

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request, Response, status

from app.clients.commands import CreateClientCommand
from app.clients.dto import ClientDTO
from app.clients.service import ClientsService
from app.containers import ApplicationContainer
from app.shared.problems import problem_responses
from app.shared.security import AdvisorId, require_service_key

router = APIRouter(
    prefix="/clients",
    tags=["clients"],
    dependencies=[Depends(require_service_key)],
    responses=problem_responses(
        status.HTTP_401_UNAUTHORIZED, status.HTTP_422_UNPROCESSABLE_CONTENT
    ),
)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a client",
    responses=problem_responses(status.HTTP_409_CONFLICT),
)
@inject
async def create_client(
    command: CreateClientCommand,
    advisor_id: AdvisorId,
    request: Request,
    response: Response,
    service: ClientsService = Depends(Provide[ApplicationContainer.clients_service]),  # noqa: B008
) -> ClientDTO:
    """Create a client owned by the calling advisor (`X-Advisor-Id`).

    The email is normalised to lower case and must be unique among the advisor's clients
    (409 otherwise). Other advisors may have a client with the same email: each advisor
    keeps their own record of a person.
    """
    client = await service.create_client(advisor_id, command)
    response.headers["Location"] = str(request.url_for("get_client", client_id=client.id))
    return client


@router.get(
    "/{client_id}",
    summary="Get a client",
    responses=problem_responses(status.HTTP_404_NOT_FOUND),
)
@inject
async def get_client(
    client_id: uuid.UUID,
    advisor_id: AdvisorId,
    service: ClientsService = Depends(Provide[ApplicationContainer.clients_service]),  # noqa: B008
) -> ClientDTO:
    """Return one of the calling advisor's clients.

    404 both when the client does not exist and when it belongs to another advisor.
    """
    return await service.get_client(advisor_id, client_id)
