"""Local embedding model (fastembed / ONNX Runtime, CPU)."""

import asyncio
import logging
import time
from collections.abc import Sequence

import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Retrieval instruction that bge models expect on the query side only.
# fastembed does not add it; see docs/embedding-experiment.md (it improves hit@1 and MRR).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-base-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-large-en-v1.5": _BGE_QUERY_PREFIX,
}


class FastEmbedEmbedder:
    """Local English embedder (bge-small-en-v1.5, ONNX) that keeps the project self-contained.

    Assumptions and trade-offs:
    - Language: the client base is UK/US, so English only is assumed. Exact language
      requirements are unknown; an English model keeps things simple. If other languages are
      needed (e.g. Chinese-speaking clients), switch to a multilingual model such as
      `intfloat/multilingual-e5-large` or `BAAI/bge-m3`, and change the full-text search
      configuration from 'english' to 'simple'.
    - Local vs hosted: a local model is used so the assignment runs with no API keys, no
      network and no cost. Hosted embedding APIs (OpenAI `text-embedding-3-*`, Cohere
      `embed-multilingual-v3`, Voyage) are expected to give better quality and also cover
      multilingual search. Even large local models miss domain facts such as "a utility bill
      is a proof of address" (docs/embedding-experiment.md), hence the domain thesaurus in
      the search module.
    - Swapping: add another `Embedder` implementation and rebind it in `containers.py`; if
      the vector size changes, add a migration and re-index (see the `embedding_model` column).
    """

    def __init__(self, model_name: str, cache_dir: str | None, threads: int | None) -> None:
        started = time.perf_counter()
        self._model = TextEmbedding(model_name, cache_dir=cache_dir, threads=threads)
        self._model_name = model_name
        self._query_prefix = QUERY_PREFIXES.get(model_name, "")
        logger.info(
            "Embedding model loaded model=%s seconds=%.1f",
            model_name,
            time.perf_counter() - started,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._model.embedding_size)

    def count_tokens(self, text: str) -> int:
        """Tokens the model sees for `text`, including special tokens (window: 512)."""
        return self._model.token_count(text)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # ONNX inference is CPU-bound: keep it off the event loop.
        return await asyncio.to_thread(self._embed, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self._embed, [self._query_prefix + text])
        return vectors[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Unit length, so that cosine similarity is well defined whatever the model returns.
        return [
            (vector / np.linalg.norm(vector)).astype(float).tolist()
            for vector in self._model.embed(texts)
        ]
