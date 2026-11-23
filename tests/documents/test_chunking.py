from itertools import pairwise

import pytest

from app.documents.chunking import split_into_chunks


def _sentences(count: int, words_per_sentence: int = 10) -> str:
    return " ".join(
        " ".join(f"s{i}w{j}" for j in range(words_per_sentence)) + "." for i in range(count)
    )


def test_split_short_text_returns_single_chunk() -> None:
    assert split_into_chunks("Utility bill for March. Amount due £187.") == [
        "Utility bill for March. Amount due £187."
    ]


def test_split_long_text_respects_word_limit_and_covers_every_word() -> None:
    text = _sentences(60)  # 600 words

    chunks = split_into_chunks(text, max_words=200, overlap_words=40)

    assert len(chunks) > 3
    assert all(len(chunk.split()) <= 200 for chunk in chunks)
    assert {w for chunk in chunks for w in chunk.split()} == set(text.split())


def test_split_consecutive_chunks_overlap_by_whole_sentences() -> None:
    chunks = split_into_chunks(_sentences(60), max_words=200, overlap_words=40)

    for previous, following in pairwise(chunks):
        first_sentence = following.split(". ")[0] + "."
        shared = set(previous.split()) & set(following.split())
        assert first_sentence in previous
        assert 0 < len(shared) <= 40


def test_split_sentence_longer_than_limit_is_cut_into_slices() -> None:
    text = " ".join(f"w{i}" for i in range(450))  # one sentence, no punctuation

    chunks = split_into_chunks(text, max_words=200, overlap_words=40)

    assert all(len(chunk.split()) <= 200 for chunk in chunks)
    assert {w for chunk in chunks for w in chunk.split()} == set(text.split())


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_split_blank_text_returns_no_chunks(text: str) -> None:
    assert split_into_chunks(text) == []
