"""The OpenAPI document is part of the contract: complete, valid examples, committed."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, PostgresDsn

from app.clients.commands import CreateClientCommand
from app.documents.commands import CreateDocumentCommand
from app.main import create_app
from app.shared.settings import Settings

OPENAPI_PATH = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


@pytest.fixture(scope="module")
def openapi() -> dict[str, Any]:
    app = create_app(Settings(database_url=PostgresDsn("postgresql+asyncpg://u:p@localhost/db")))
    schema: dict[str, Any] = app.openapi()
    return schema


def test_every_operation_has_a_summary(openapi: dict[str, Any]) -> None:
    missing = [
        f"{method.upper()} {path}"
        for path, operations in openapi["paths"].items()
        for method, operation in operations.items()
        if method in HTTP_METHODS and not operation.get("summary")
    ]
    assert missing == []


@pytest.mark.parametrize("model", [CreateClientCommand, CreateDocumentCommand])
def test_request_body_examples_are_valid(model: type[BaseModel]) -> None:
    examples = model.model_json_schema()["examples"]

    assert examples
    for example in examples:
        model.model_validate(example)


def test_committed_openapi_document_is_up_to_date(openapi: dict[str, Any]) -> None:
    committed = json.loads(OPENAPI_PATH.read_text())

    assert committed == openapi, "run `make openapi` to refresh docs/openapi.json"
