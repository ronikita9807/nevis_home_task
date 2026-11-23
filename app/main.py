"""Application factory and ASGI entrypoint."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Build the FastAPI application: routers, handlers and lifecycle hooks."""
    app = FastAPI(title="Nevis Search API", version="0.1.0")

    @app.get("/health", tags=["service"], summary="Liveness probe")
    def health() -> dict[str, str]:
        """Return 200 while the process is alive. Dependencies are not checked."""
        return {"status": "ok"}

    return app


app = create_app()
