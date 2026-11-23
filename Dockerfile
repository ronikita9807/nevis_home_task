# syntax=docker/dockerfile:1

# --- Build stage: resolve and install dependencies into a virtualenv ----------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.23 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /srv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Ship the embedding model in the image: no download at startup, works offline.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN /srv/.venv/bin/python -c \
    "import sys; from fastembed import TextEmbedding; TextEmbedding(sys.argv[1], cache_dir='/srv/models')" \
    "$EMBEDDING_MODEL"

# --- Runtime stage: only the virtualenv and the application code --------------
FROM python:3.13-slim

RUN useradd --system --create-home --uid 10001 app

WORKDIR /srv

COPY --from=builder /srv/.venv /srv/.venv
COPY --from=builder --chown=app:app /srv/models /srv/models
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
# Demo data and its loader: `docker compose exec api python scripts/seed.py`
COPY data ./data
COPY scripts/seed.py ./scripts/seed.py

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV PATH="/srv/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=2 \
    EMBEDDING_MODEL=$EMBEDDING_MODEL \
    EMBEDDING_CACHE_DIR=/srv/models \
    HF_HUB_OFFLINE=1

USER app
EXPOSE 8000

# Indexing runs inside the request (~0.2 s per KB of text on a laptop, ~20 s for the 100 KB
# maximum): the worker timeout must exceed it (gunicorn's default is 30 s).
CMD ["sh", "-c", "exec gunicorn 'app.main:create_app()' --worker-class uvicorn_worker.UvicornWorker --workers ${WEB_CONCURRENCY} --timeout 120 --graceful-timeout 30 --bind 0.0.0.0:8000"]
