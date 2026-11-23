"""Shared fixtures: a real Postgres (pgvector) container, migrated with Alembic."""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from dependency_injector import providers
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import PostgresDsn
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.main import create_app
from app.shared.db import Base
from app.shared.settings import Settings
from tests.documents.fakes import FakeEmbedder

POSTGRES_IMAGE = "pgvector/pgvector:0.8.2-pg17"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    # The reaper container mounts the Docker socket. Docker Desktop (macOS) exposes it under
    # the user's home, which cannot be mounted; the in-VM path works there and on Linux CI.
    # testcontainers reads its config at import time, hence the import below.
    os.environ.setdefault("TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE", "/var/run/docker.sock")
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(POSTGRES_IMAGE, driver="asyncpg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture(scope="session")
def migrated_database_url(database_url: str) -> str:
    alembic_config = Config(PROJECT_ROOT / "alembic.ini")
    alembic_config.attributes["configure_logger"] = False
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        command.upgrade(alembic_config, "head")
    finally:
        if previous is None:
            del os.environ["DATABASE_URL"]
        else:
            os.environ["DATABASE_URL"] = previous
    return database_url


@pytest.fixture
async def clean_database_url(migrated_database_url: str) -> str:
    """The migrated database with every table emptied: tests do not see each other's rows."""
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    engine = create_async_engine(migrated_database_url, poolclass=NullPool)
    async with engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {tables} CASCADE"))
    await engine.dispose()
    return migrated_database_url


@pytest.fixture
def settings(clean_database_url: str) -> Settings:
    # No real LLM in tests, even if the developer's environment has ANTHROPIC_API_KEY set.
    return Settings(database_url=PostgresDsn(clean_database_url), anthropic_api_key=None)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
async def app(settings: Settings, fake_embedder: FakeEmbedder) -> AsyncIterator[FastAPI]:
    """The application with the fake embedder; tests of real semantics use their own app."""
    application = create_app(settings)
    application.state.container.embedder.override(providers.Object(fake_embedder))
    yield application
    await application.state.container.database().dispose()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
