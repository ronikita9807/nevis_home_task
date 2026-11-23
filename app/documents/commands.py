"""Write-side inputs of the documents module (request bodies as defined by the spec)."""

from pydantic import ConfigDict

from app.documents.domain import Content, Title
from app.shared.domain import BaseCommand


class CreateDocumentCommand(BaseCommand):
    """Body of `POST /clients/{id}/documents`. The client comes from the path."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "title": "Utility bill — British Gas, March 2024",
                    "content": "Energy statement for 14 Elm Road, Richmond. Amount due: £187.42.",
                }
            ]
        }
    )

    title: Title
    content: Content
