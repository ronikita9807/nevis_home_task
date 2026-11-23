"""Application factory and ASGI entrypoint.

Run with a factory, e.g. `uvicorn --factory app.main:create_app`, so that settings are
read and validated at startup rather than at import time.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.containers import ApplicationContainer
from app.shared import health
from app.shared.observability import RequestContextMiddleware, configure_logging
from app.shared.problems import register_problem_handlers
from app.shared.settings import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application: container, routers, handlers and lifecycle hooks."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    container = ApplicationContainer()
    container.config.from_dict(settings.model_dump(mode="json"))
    container.wire(modules=[health])

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        yield
        await container.database().dispose()

    app = FastAPI(title="Nevis Search API", version="0.1.0", lifespan=lifespan)
    app.state.container = container

    app.add_middleware(RequestContextMiddleware)
    register_problem_handlers(app)
    app.include_router(health.router)
    return app
