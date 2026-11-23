"""Splitting document text into overlapping chunks for embedding.

The embedding model sees at most 512 tokens; a whole long document squeezed into one vector
also blurs its meaning. Chunks of ~200 words (~260-300 tokens) stay well inside the window
and keep one topic each; the overlap keeps a sentence that straddles a boundary searchable.
"""

import re

CHUNK_MAX_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

type Unit = list[str]  # the words of one sentence (or of a slice of a very long sentence)


def split_into_chunks(
    text: str,
    max_words: int = CHUNK_MAX_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split text into chunks of whole sentences, each at most `max_words` words.

    Consecutive chunks share trailing sentences of up to `overlap_words` words.
    Sentences longer than `max_words` are cut into word slices.
    """
    chunks: list[list[Unit]] = []
    current: list[Unit] = []
    for unit in _units(text, max_words):
        if current and _word_count(current) + len(unit) > max_words:
            chunks.append(current)
            current = _overlap_tail(current, overlap_words)
            while current and _word_count(current) + len(unit) > max_words:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append(current)
    return [" ".join(word for unit in chunk for word in unit) for chunk in chunks]


def _units(text: str, max_words: int) -> list[Unit]:
    units: list[Unit] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        words = sentence.split()
        units.extend(words[i : i + max_words] for i in range(0, len(words), max_words))
    return units


def _overlap_tail(units: list[Unit], overlap_words: int) -> list[Unit]:
    tail: list[Unit] = []
    for unit in reversed(units):
        if _word_count(tail) + len(unit) > overlap_words:
            break
        tail.insert(0, unit)
    return tail


def _word_count(units: list[Unit]) -> int:
    return sum(len(unit) for unit in units)
