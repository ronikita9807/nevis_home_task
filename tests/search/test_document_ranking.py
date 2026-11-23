import uuid

from app.search.domain import KeywordMatch, MatchChannel, SemanticMatch
from app.search.ranking import fuse_document_rankings, truncate_words
from app.search.thesaurus import expand_query

A, B, C = (uuid.UUID(int=i) for i in (1, 2, 3))


def test_expand_query_maps_address_proof_to_document_types() -> None:
    expansions = expand_query("address proof")

    assert {"utility bill", "council tax bill", "tenancy agreement"} <= set(expansions)
    assert "address proof" not in expansions


def test_expand_query_matches_whole_words_only() -> None:
    assert expand_query("passports") == []
    assert "driving licence" in expand_query("copy of passport")


def test_expand_query_unrelated_text_has_no_expansions() -> None:
    assert expand_query("what was discussed last summer") == []


def test_fuse_document_found_by_both_channels_ranks_first() -> None:
    semantic = [SemanticMatch(A, 0.7, "a"), SemanticMatch(B, 0.6, "b")]
    keyword = [KeywordMatch(C, 0.9, "c"), KeywordMatch(B, 0.5, "b headline")]

    fused = fuse_document_rankings(semantic, keyword)

    assert fused[0].document_id == B
    assert fused[0].channels == [MatchChannel.SEMANTIC, MatchChannel.KEYWORD]
    assert fused[0].snippet == "b headline"  # keyword headline shows the matched words
    assert {d.document_id for d in fused} == {A, B, C}


def test_fuse_keyword_match_outranks_semantic_match_at_same_rank() -> None:
    fused = fuse_document_rankings([SemanticMatch(A, 0.9, "a")], [KeywordMatch(B, 0.1, "b")])

    assert [d.document_id for d in fused] == [B, A]


def test_fuse_empty_channels_returns_nothing() -> None:
    assert fuse_document_rankings([], []) == []


def test_truncate_words_adds_ellipsis_only_when_cut() -> None:
    assert truncate_words("one two three", 3) == "one two three"
    assert truncate_words("one two three four", 3) == "one two three …"
