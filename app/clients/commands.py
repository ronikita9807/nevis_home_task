"""Write-side inputs of the clients module (request bodies as defined by the spec)."""

from pydantic import ConfigDict, Field

from app.clients.domain import Description, NormalizedEmail, PersonName, SocialLinks
from app.shared.domain import BaseCommand


class CreateClientCommand(BaseCommand):
    """Body of `POST /clients`. The owning advisor comes from `X-Advisor-Id`, not the body."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "first_name": "John",
                    "last_name": "Doe",
                    "email": "john.doe@neviswealth.com",
                    "description": "Senior software engineer, long-term growth investor.",
                    "social_links": ["https://www.linkedin.com/in/john-doe-example"],
                }
            ]
        }
    )

    first_name: PersonName
    last_name: PersonName
    email: NormalizedEmail
    description: Description | None = None
    social_links: SocialLinks = Field(default_factory=list[str])
