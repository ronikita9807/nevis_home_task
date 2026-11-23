import uuid

import pytest

from app.search.domain import (
    ClientCandidate,
    ClientField,
    MatchChannel,
    SemanticClientMatch,
    normalize_query,
)
from app.search.infra.queries import escape_like
from app.search.ranking import (
    EXACT_SCORE,
    PREFIX_SCORE,
    SUBSTRING_SCORE,
    edit_distance,
    is_typo_of,
    merge_semantic_clients,
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


def test_rank_clients_fragment_of_a_longer_word_is_not_a_typo() -> None:
    wealthy = _candidate("Liam", "Stone", "liam@law.ie", "Passing wealth to children", 0.42)

    assert rank_clients("neviswealth", [wealthy]) == []


@pytest.mark.parametrize(
    ("word", "text", "expected"),
    [
        ("jonh", "John", True),
        ("cartre", "emily carter", True),
        ("mueller", "Müller", True),
        ("investr", "growth investor", True),
        ("neviswealth", "passing wealth to his children", False),
        ("smith", "john", False),
    ],
)
def test_is_typo_of(word: str, text: str, expected: bool) -> None:
    assert is_typo_of(word, text.lower()) is expected


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("john", "john", 0), ("jonh", "john", 1), ("patell", "patel", 1), ("abc", "", 3)],
)
def test_edit_distance(a: str, b: str, distance: int) -> None:
    assert edit_distance(a, b) == distance


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


def _semantic(candidate: ClientCandidate, similarity: float) -> SemanticClientMatch:
    return SemanticClientMatch(candidate=candidate, similarity=similarity)


def test_merge_semantic_clients_keeps_only_near_best_above_floor() -> None:
    liam, emily, raj = (_candidate(first=n, email=f"{n}@x.com") for n in ("Liam", "Emily", "Raj"))
    semantic = [_semantic(liam, 0.574), _semantic(emily, 0.560), _semantic(raj, 0.492)]

    merged = merge_semantic_clients([], semantic)

    assert [r.candidate for r in merged] == [liam, emily]  # raj is outside the 0.03 band
    assert all(r.channels == (MatchChannel.SEMANTIC,) for r in merged)
    assert merged[0].matched_fields == [ClientField.DESCRIPTION]


def test_merge_semantic_clients_below_floor_adds_nothing() -> None:
    assert merge_semantic_clients([], [_semantic(_candidate(), 0.45)]) == []


def test_merge_semantic_clients_ranks_meaning_below_word_matches() -> None:
    by_word = _candidate(first="Emily", email="emily@x.com", description="cautious investor")
    by_meaning = _candidate(first="Liam", email="liam@x.com")
    [word_match] = rank_clients("cautious", [by_word])

    merged = merge_semantic_clients([word_match], [_semantic(by_meaning, 0.65)])

    assert [r.candidate for r in merged] == [by_word, by_meaning]
    assert merged[1].score < SUBSTRING_SCORE


def test_merge_semantic_clients_found_both_ways_keeps_word_score() -> None:
    client = _candidate(description="cautious investor")
    [word_match] = rank_clients("cautious", [client])

    [merged] = merge_semantic_clients([word_match], [_semantic(client, 0.6)])

    assert merged.score == word_match.score
    assert merged.channels == (MatchChannel.KEYWORD, MatchChannel.SEMANTIC)
