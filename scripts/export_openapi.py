"""Write the OpenAPI document to docs/openapi.json (checked by tests/test_openapi.py).

Usage: uv run python scripts/export_openapi.py
"""

import json
from pathlib import Path

from pydantic import PostgresDsn

from app.main import create_app
from app.shared.settings import Settings

OPENAPI_PATH = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"
# Building the schema never connects to the database.
_PLACEHOLDER_DB = PostgresDsn("postgresql+asyncpg://user:password@localhost:5432/db")


def render_openapi() -> str:
    app = create_app(Settings(database_url=_PLACEHOLDER_DB))
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OPENAPI_PATH.write_text(render_openapi())
    print(f"written {OPENAPI_PATH}")
