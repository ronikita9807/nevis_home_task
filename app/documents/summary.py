"""Document summaries: an LLM when configured, an extractive summary otherwise or on failure."""

import logging
import uuid
from collections.abc import Callable, Sequence

from app.documents.chunking import split_sentences
from app.documents.domain import DocumentSummary, SummaryMethod
from app.documents.dto import SummaryDTO
from app.documents.exceptions import SummarizationError
from app.documents.ports import DocumentsUnitOfWork, Summarizer
from app.shared.embeddings import Embedder

logger = logging.getLogger(__name__)

SUMMARY_SENTENCES = 3
# Bounds the extra inference for very long documents; the opening usually carries the gist.
MAX_SENTENCES_CONSIDERED = 200


def pick_central_sentences(
    sentences: Sequence[str], vectors: Sequence[Sequence[float]], count: int
) -> list[str]:
    """The `count` sentences closest to the document's mean embedding, in original order."""
    dimension = len(vectors[0])
    centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dimension)]
    scores = [sum(a * b for a, b in zip(v, centroid, strict=True)) for v in vectors]
    best = sorted(range(len(sentences)), key=lambda i: -scores[i])[:count]
    return [sentences[i] for i in sorted(best)]


class ExtractiveSummarizer:
    """Picks the most representative sentences. Local, deterministic, no data leaves."""

    method = SummaryMethod.EXTRACTIVE

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def summarize(self, title: str, content: str) -> str:
        sentences = split_sentences(content)[:MAX_SENTENCES_CONSIDERED]
        if len(sentences) <= SUMMARY_SENTENCES:
            return " ".join(sentences)
        vectors = await self._embedder.embed_documents(sentences)
        return " ".join(pick_central_sentences(sentences, vectors, SUMMARY_SENTENCES))


class SummaryService:
    def __init__(
        self,
        uow_factory: Callable[[], DocumentsUnitOfWork],
        extractive: Summarizer,
        llm: Summarizer | None,
    ) -> None:
        self._uow_factory = uow_factory
        self._extractive = extractive
        self._llm = llm

    async def get_summary(self, advisor_id: uuid.UUID, document_id: uuid.UUID) -> SummaryDTO:
        async with self._uow_factory() as uow:
            document = await uow.documents.get(advisor_id, document_id)
        if document.summary is not None:
            return SummaryDTO.build(document_id, document.summary)

        # Generated outside any transaction: an LLM call can take seconds.
        # Trade-off: a fallback (extractive) summary is stored too, so a one-off LLM failure
        # keeps it for this document; production would retry LLM summaries later.
        summary = await self._summarize(document.title, document.content)
        async with self._uow_factory() as uow:
            stored = await uow.documents.save_summary_if_absent(advisor_id, document_id, summary)
            await uow.commit()
        return SummaryDTO.build(document_id, stored)

    async def _summarize(self, title: str, content: str) -> DocumentSummary:
        if self._llm is not None:
            try:
                text = await self._llm.summarize(title, content)
            except SummarizationError as exc:
                # The feature degrades instead of failing the request.
                logger.warning("LLM summary failed, using extractive summary: %s", exc.detail)
            else:
                return DocumentSummary(text=text, method=self._llm.method)
        text = await self._extractive.summarize(title, content)
        return DocumentSummary(text=text, method=SummaryMethod.EXTRACTIVE)
