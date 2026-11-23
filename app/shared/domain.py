"""Base classes for domain models, commands and read models."""

from pydantic import BaseModel, ConfigDict


class BaseDomainModel(BaseModel):
    """Domain entity: invariants live in types and validators, re-checked on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BaseCommand(BaseModel):
    """Write-side input (also the API request body). Immutable, unknown fields rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseDTO(BaseModel):
    """Read-side output (also the API response body). Immutable."""

    model_config = ConfigDict(frozen=True)
