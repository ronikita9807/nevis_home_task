from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.main import create_app


async def test_health_when_app_started_returns_ok() -> None:
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}
