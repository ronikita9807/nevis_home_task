from app.documents.summary import ExtractiveSummarizer, pick_central_sentences
from tests.documents.fakes import FakeEmbedder


def test_pick_central_sentences_keeps_original_order() -> None:
    sentences = ["off topic", "pension a", "pension b", "pension c"]
    vectors = [[0.0, 1.0], [1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]

    assert pick_central_sentences(sentences, vectors, 2) == ["pension a", "pension b"]


async def test_extractive_summary_of_short_document_is_the_document() -> None:
    summarizer = ExtractiveSummarizer(FakeEmbedder())

    text = await summarizer.summarize("Note", "First fact. Second fact.")

    assert text == "First fact. Second fact."


async def test_extractive_summary_of_long_document_has_three_of_its_sentences() -> None:
    sentences = [f"Sentence {i} about the pension plan." for i in range(10)]

    text = await ExtractiveSummarizer(FakeEmbedder()).summarize("Plan", " ".join(sentences))

    picked = [s for s in sentences if s in text]
    assert len(picked) == 3
