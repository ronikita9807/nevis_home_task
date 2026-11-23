"""Application factory and ASGI entrypoint.

Run with a factory, e.g. `uvicorn --factory app.main:create_app`, so that settings are
read and validated at startup rather than at import time.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients import api as clients_api
from app.containers import ApplicationContainer
from app.documents import api as documents_api
from app.documents.domain import EMBEDDING_DIMENSION
from app.search import api as search_api
from app.shared import health, security
from app.shared.observability import RequestContextMiddleware, configure_logging
from app.shared.problems import register_problem_handlers
from app.shared.settings import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application: container, routers, handlers and lifecycle hooks."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    container = ApplicationContainer()
    container.config.from_dict(settings.model_dump(mode="json", exclude={"api_key"}))
    # Passed separately: a JSON dump masks secrets as "**********".
    api_key = settings.api_key.get_secret_value() if settings.api_key else None
    container.config.api_key.from_value(api_key)
    container.wire(modules=[health, security, clients_api, documents_api, search_api])

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        # Load the model before serving: the first request must not pay for it.
        embedder = container.embedder()
        if embedder.dimension != EMBEDDING_DIMENSION:
            msg = (
                f"Embedding model {embedder.model_name} produces {embedder.dimension}-d vectors, "
                f"the database stores {EMBEDDING_DIMENSION}-d: a migration and re-index are needed"
            )
            raise RuntimeError(msg)
        yield
        await container.database().dispose()

    app = FastAPI(title="Nevis Search API", version="0.1.0", lifespan=lifespan)
    app.state.container = container

    app.add_middleware(RequestContextMiddleware)
    register_problem_handlers(app)
    app.include_router(health.router)
    app.include_router(clients_api.router)
    app.include_router(documents_api.router)
    app.include_router(search_api.router)
    return app
