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
from app.search import api as search_api
from app.shared import health, security
from app.shared.embeddings import EMBEDDING_DIMENSION
from app.shared.observability import RequestContextMiddleware, configure_logging
from app.shared.problems import register_problem_handlers
from app.shared.settings import Settings, load_settings

API_DESCRIPTION = """
Search across an advisor's clients and their documents.

**Authentication** (use *Authorize*):
- `X-Advisor-Id` — required on every business endpoint: UUID of the advisor on whose behalf
  the call is made. All data is scoped to it; another advisor's data is never returned
  (it is reported as 404).
- `X-API-Key` — shared key of the trusted caller (gateway/BFF); required only when the
  server is configured with `API_KEY`.

**Errors** use RFC 9457 Problem Details (`application/problem+json`) with a stable `code`
and the `request_id` of the call.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application: container, routers, handlers and lifecycle hooks."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    container = ApplicationContainer()
    secrets = {"api_key", "anthropic_api_key"}
    container.config.from_dict(settings.model_dump(mode="json", exclude=secrets))
    # Passed separately: a JSON dump masks secrets as "**********".
    for name in secrets:
        secret = getattr(settings, name)
        getattr(container.config, name).from_value(secret.get_secret_value() if secret else None)
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

    app = FastAPI(
        title="Nevis Search API",
        version="0.1.0",
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.container = container

    app.add_middleware(RequestContextMiddleware)
    register_problem_handlers(app)
    app.include_router(health.router)
    app.include_router(clients_api.router)
    app.include_router(documents_api.router)
    app.include_router(search_api.router)
    return app
