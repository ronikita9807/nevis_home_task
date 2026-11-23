import uuid

import pytest

from app.search.domain import ClientCandidate, ClientField, normalize_query
from app.search.infra.queries import escape_like
from app.search.ranking import (
    EXACT_SCORE,
    PREFIX_SCORE,
    SUBSTRING_SCORE,
    rank_clients,
)


def _candidate(
    first: str = "John",
    last: str = "Doe",
    email: str = "john.doe@neviswealth.com",
    description: str | None = None,
    similarity: float = 0.0,
) -> ClientCandidate:
    return ClientCandidate(
        id=uuid.uuid4(),
        first_name=first,
        last_name=last,
        email=email,
        description=description,
        social_links=[],
        similarity=dict.fromkeys(ClientField, similarity),
    )


@pytest.mark.parametrize(
    ("query", "expected_score"),
    [
        ("john.doe@neviswealth.com", EXACT_SCORE),
        ("john doe", EXACT_SCORE),
        ("doe john", EXACT_SCORE),
        ("joh", PREFIX_SCORE),
        ("neviswealth", SUBSTRING_SCORE),
        ("doe wealth", SUBSTRING_SCORE),
    ],
)
def test_rank_clients_assigns_tier_scores(query: str, expected_score: float) -> None:
    [ranked] = rank_clients(query, [_candidate()])

    assert ranked.score == expected_score


def test_rank_clients_typo_only_match_scores_below_substring() -> None:
    [ranked] = rank_clients("jonh", [_candidate(similarity=0.4)])

    assert 0 < ranked.score < SUBSTRING_SCORE


def test_rank_clients_orders_exact_before_substring_and_reports_matched_fields() -> None:
    by_description = _candidate("Emily", "Carter", "emily@carter.co.uk", "Works with John Doe")
    by_name = _candidate()

    ranked = rank_clients("john doe", [by_description, by_name])

    assert [r.candidate for r in ranked] == [by_name, by_description]
    assert ranked[1].matched_fields == [ClientField.DESCRIPTION]


def test_normalize_query_lowercases_and_collapses_whitespace() -> None:
    assert normalize_query("  John \t  DOE \n") == "john doe"


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [("50%", "50\\%"), ("a_b", "a\\_b"), ("back\\slash", "back\\\\slash"), ("plain", "plain")],
)
def test_escape_like_makes_wildcards_literal(raw: str, escaped: str) -> None:
    assert escape_like(raw) == escaped
