import pytest
from pydantic import PostgresDsn, ValidationError

from app.shared.settings import Settings

NEON_URL = (
    "postgresql://owner:secret@ep-x.eu-central-1.aws.neon.tech/nevis"
    "?sslmode=require&channel_binding=require"
)


def _database_url(url: str) -> str:
    return str(Settings(database_url=PostgresDsn(url)).database_url)


def test_database_url_from_hosted_provider_is_converted_for_asyncpg() -> None:
    assert _database_url(NEON_URL) == (
        "postgresql+asyncpg://owner:secret@ep-x.eu-central-1.aws.neon.tech/nevis?ssl=require"
    )


def test_database_url_with_asyncpg_scheme_is_kept() -> None:
    url = "postgresql+asyncpg://nevis:nevis@localhost:5433/nevis"

    assert _database_url(url) == url


def test_database_url_with_other_driver_is_rejected() -> None:
    with pytest.raises(ValidationError, match="asyncpg"):
        Settings(database_url=PostgresDsn("postgresql+psycopg://nevis:nevis@localhost/nevis"))
