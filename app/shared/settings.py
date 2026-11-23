"""Application settings, validated once at startup (fail fast on misconfiguration)."""

from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    database_url: PostgresDsn
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    # Shorter than a typical client timeout: fail with 503 instead of hanging.
    db_pool_timeout_seconds: float = Field(default=5.0, gt=0)
    db_pool_recycle_seconds: int = Field(default=1800, gt=0)

    # Shared service key of the trusted caller (gateway/BFF). When unset (local runs),
    # the key is not checked; advisor identity is still required on every business call.
    api_key: SecretStr | None = None

    log_level: LogLevel = "INFO"

    # Chosen in docs/embedding-experiment.md. Its size must match the `vector(384)` column.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where model weights are cached; the Docker image ships them, so no download at runtime.
    embedding_cache_dir: str | None = None
    # ONNX Runtime threads per process (None = all cores); keep it low with several workers.
    embedding_threads: int | None = Field(default=None, ge=1)

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_api_key_means_unset(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, value: PostgresDsn) -> PostgresDsn:
        if value.scheme != "postgresql+asyncpg":
            msg = "DATABASE_URL must use the 'postgresql+asyncpg' scheme"
            raise ValueError(msg)
        return value


def load_settings() -> Settings:
    """Read settings from the environment (and `.env`); raises on invalid configuration."""
    # Required fields are filled from the environment, which pyright cannot see.
    return Settings()  # pyright: ignore[reportCallIssue]
