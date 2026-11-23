"""HTTP endpoint of the search module."""

from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Query, status

from app.containers import ApplicationContainer
from app.search.domain import DEFAULT_LIMIT, MAX_LIMIT, SearchText, SearchType
from app.search.dto import SearchHit
from app.search.service import SearchService
from app.shared.problems import problem_responses
from app.shared.security import AdvisorId, require_service_key

router = APIRouter(
    tags=["search"],
    dependencies=[Depends(require_service_key)],
    responses=problem_responses(
        status.HTTP_401_UNAUTHORIZED, status.HTTP_422_UNPROCESSABLE_CONTENT
    ),
)


@router.get("/search", summary="Search clients and documents")
@inject
async def search(
    q: Annotated[SearchText, Query(description="Search text (1-256 characters)")],
    advisor_id: AdvisorId,
    type: Annotated[SearchType | None, Query(description="Return only one kind of result")] = None,
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description="Maximum results per kind")
    ] = DEFAULT_LIMIT,
    service: SearchService = Depends(Provide[ApplicationContainer.search_service]),  # noqa: B008
) -> list[SearchHit]:
    """Search the calling advisor's clients and documents.

    **Clients** match on email, first/last name and description: every query word must
    appear in some field (case-insensitive), with tolerance for typos in words of 4+ letters.
    Example: `NevisWealth` finds `john.doe@neviswealth.com`.

    **Documents** are found by meaning (embeddings of document chunks) and by keywords,
    including known synonyms of document types. Example: `address proof` finds documents
    containing `utility bill`. Each hit has a snippet and the owning client.

    The response is an array: client hits first, then document hits, each block ordered by
    its own `score`. Only the calling advisor's data is searched.
    """
    return await service.search(advisor_id, q, type, limit)
