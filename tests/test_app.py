from fastapi import status
from httpx import ASGITransport, AsyncClient
from pydantic import PostgresDsn

from app.main import create_app
from app.shared.problems import PROBLEM_MEDIA_TYPE
from app.shared.settings import Settings

# Nothing listens on port 1: connection is refused immediately.
UNREACHABLE_DB_URL = PostgresDsn("postgresql+asyncpg://user:pass@127.0.0.1:1/db")


async def test_health_when_app_started_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


async def test_ready_when_database_reachable_returns_ready(client: AsyncClient) -> None:
    response = await client.get("/ready")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ready"}


async def test_ready_when_database_unreachable_returns_503_problem() -> None:
    app = create_app(Settings(database_url=UNREACHABLE_DB_URL))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["code"] == "service_unavailable"


async def test_unknown_route_returns_404_problem_with_request_id(client: AsyncClient) -> None:
    response = await client.get("/missing", headers={"X-Request-ID": "req-123"})

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.headers["X-Request-ID"] == "req-123"
    assert response.json() == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "detail": "Not Found",
        "instance": "/missing",
        "code": "not_found",
        "request_id": "req-123",
    }
