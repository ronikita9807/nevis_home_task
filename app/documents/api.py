"""HTTP endpoints of the documents module."""

import uuid

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request, Response, status

from app.containers import ApplicationContainer
from app.documents.commands import CreateDocumentCommand
from app.documents.dto import DocumentDTO, SummaryDTO
from app.documents.service import DocumentsService
from app.documents.summary import SummaryService
from app.shared.problems import problem_responses
from app.shared.security import AdvisorId, require_service_key

router = APIRouter(
    tags=["documents"],
    dependencies=[Depends(require_service_key)],
    responses=problem_responses(
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_404_NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
    ),
)


@router.post(
    "/clients/{client_id}/documents",
    status_code=status.HTTP_201_CREATED,
    summary="Add a document to a client",
)
@inject
async def create_document(
    client_id: uuid.UUID,
    command: CreateDocumentCommand,
    advisor_id: AdvisorId,
    request: Request,
    response: Response,
    service: DocumentsService = Depends(Provide[ApplicationContainer.documents_service]),  # noqa: B008
) -> DocumentDTO:
    """Store a document for one of the calling advisor's clients and index it for search.

    The content is split into chunks and embedded synchronously, so the document is
    searchable as soon as this returns. 404 if the client does not exist or belongs to
    another advisor. Not idempotent: retrying the request stores a second document.
    """
    document = await service.create_document(advisor_id, client_id, command)
    response.headers["Location"] = str(request.url_for("get_document", document_id=document.id))
    return document


@router.get("/documents/{document_id}", summary="Get a document")
@inject
async def get_document(
    document_id: uuid.UUID,
    advisor_id: AdvisorId,
    service: DocumentsService = Depends(Provide[ApplicationContainer.documents_service]),  # noqa: B008
) -> DocumentDTO:
    """Return one of the calling advisor's documents (404 for another advisor's document)."""
    return await service.get_document(advisor_id, document_id)


@router.get("/documents/{document_id}/summary", summary="Get a short summary of a document")
@inject
async def get_document_summary(
    document_id: uuid.UUID,
    advisor_id: AdvisorId,
    service: SummaryService = Depends(Provide[ApplicationContainer.summary_service]),  # noqa: B008
) -> SummaryDTO:
    """Return a 2-3 sentence summary of one of the calling advisor's documents.

    Written by Claude when the server is configured with ANTHROPIC_API_KEY (`method: llm`),
    otherwise - or if the model fails or times out - the most representative sentences of the
    document are used (`method: extractive`). Computed on the first request, then cached.
    """
    return await service.get_summary(advisor_id, document_id)
