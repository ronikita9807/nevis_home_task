"""Client entity and the field types shared by its command and domain model."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Self

from pydantic import AfterValidator, EmailStr, Field, StringConstraints

from app.shared.domain import BaseDomainModel

MAX_NAME_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 5_000
MAX_SOCIAL_LINKS = 10
MAX_SOCIAL_LINK_LENGTH = 2_048

PersonName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NAME_LENGTH)
]
# The whole address is lower-cased: `John@X.com` and `john@x.com` are one client.
NormalizedEmail = Annotated[EmailStr, AfterValidator(str.lower)]
Description = Annotated[
    str, StringConstraints(strip_whitespace=True, max_length=MAX_DESCRIPTION_LENGTH)
]
# Free-form strings as in the spec (no URL format is imposed), only bounded.
SocialLink = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_SOCIAL_LINK_LENGTH)
]
SocialLinks = Annotated[list[SocialLink], Field(max_length=MAX_SOCIAL_LINKS)]


class Client(BaseDomainModel):
    """A client record. Owned by exactly one advisor; never visible to other advisors."""

    id: uuid.UUID
    advisor_id: uuid.UUID
    first_name: PersonName
    last_name: PersonName
    email: NormalizedEmail
    description: Description | None
    social_links: SocialLinks
    created_at: datetime

    @classmethod
    def create(
        cls,
        *,
        advisor_id: uuid.UUID,
        first_name: str,
        last_name: str,
        email: str,
        description: str | None,
        social_links: list[str],
    ) -> Self:
        return cls(
            id=uuid.uuid4(),
            advisor_id=advisor_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            description=description or None,
            social_links=social_links,
            created_at=datetime.now(UTC),
        )
