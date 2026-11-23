"""Test helpers shared across modules."""

import uuid

ADVISOR_A = uuid.UUID("a1a1a1a1-0000-4000-8000-000000000001")
ADVISOR_B = uuid.UUID("b2b2b2b2-0000-4000-8000-000000000002")


def as_advisor(advisor_id: uuid.UUID) -> dict[str, str]:
    return {"X-Advisor-Id": str(advisor_id)}
