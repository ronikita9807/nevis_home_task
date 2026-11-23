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

# --- Runtime stage: only the virtualenv and the application code --------------
FROM python:3.13-slim

RUN useradd --system --create-home --uid 10001 app

WORKDIR /srv

COPY --from=builder /srv/.venv /srv/.venv
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app

ENV PATH="/srv/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=2

USER app
EXPOSE 8000

CMD ["sh", "-c", "exec gunicorn 'app.main:create_app()' --worker-class uvicorn_worker.UvicornWorker --workers ${WEB_CONCURRENCY} --bind 0.0.0.0:8000"]
