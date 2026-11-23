"""Compare local embedding models on the demo corpus and golden queries.

Usage: uv run python scripts/embedding_experiment.py [--extended]

`--extended` also evaluates larger embedding models and cross-encoder rerankers
(downloads several GB of model weights).

Prints markdown tables: retrieval quality (hit@1, recall@3/5, MRR), score separation
between relevant and irrelevant documents (to pick a similarity cut-off), latency, and the
canonical pairs from the assignment. Documents are embedded whole (title + content): every
demo document fits the 512-token window, so chunking does not affect this comparison.
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from numpy.typing import NDArray

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

type Vector = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Variant:
    label: str
    model: str
    query_prefix: str = ""
    document_prefix: str = ""


VARIANTS = [
    Variant("bge-small-en-v1.5 + query prefix", "BAAI/bge-small-en-v1.5", BGE_QUERY_PREFIX),
    Variant("bge-small-en-v1.5, no prefix", "BAAI/bge-small-en-v1.5"),
    Variant("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
    Variant(
        "arctic-embed-s + query prefix", "snowflake/snowflake-arctic-embed-s", BGE_QUERY_PREFIX
    ),
    Variant("bge-base-en-v1.5 + query prefix (768d)", "BAAI/bge-base-en-v1.5", BGE_QUERY_PREFIX),
]

EXTENDED_VARIANTS = [
    Variant("bge-large-en-v1.5 + query prefix (1024d)", "BAAI/bge-large-en-v1.5", BGE_QUERY_PREFIX),
    Variant(
        "mxbai-embed-large-v1 + query prefix (1024d)",
        "mixedbread-ai/mxbai-embed-large-v1",
        BGE_QUERY_PREFIX,
    ),
    Variant(
        "nomic-embed-text-v1.5 (768d)",
        "nomic-ai/nomic-embed-text-v1.5",
        "search_query: ",
        "search_document: ",
    ),
]
RERANKERS = [
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-v2-m3-int8",
]

# The assignment's headline case and the documents a human expects it to find.
ADDRESS_PROOF_QUERY = "address proof"
ADDRESS_PROOF_DOCS = ["john_utility_bill", "emily_council_tax", "sophie_tenancy"]

CANONICAL_QUERY = "address proof"
CANONICAL_TEXTS = ["utility bill", "council tax bill", "bank statement", "stock portfolio"]


@dataclass(frozen=True, slots=True)
class Document:
    key: str
    advisor: str
    text: str


@dataclass(frozen=True, slots=True)
class Query:
    advisor: str
    text: str
    relevant: frozenset[str]


@dataclass(frozen=True, slots=True)
class Result:
    variant: Variant
    hit_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    relevant_scores: list[float]
    irrelevant_scores: list[float]
    top1_irrelevant_scores: list[float]
    rankings: list[list[tuple[float, str]]]
    load_seconds: float
    ms_per_document: float
    canonical: dict[str, float]


def load_corpus() -> tuple[list[Document], list[Query]]:
    corpus = json.loads((DATA_DIR / "demo_corpus.json").read_text())
    documents = [
        Document(
            key=doc["key"], advisor=client["advisor"], text=f"{doc['title']}\n{doc['content']}"
        )
        for client in corpus["clients"]
        for doc in client["documents"]
    ]
    golden = json.loads((DATA_DIR / "golden_queries.json").read_text())
    queries = [
        Query(advisor=q["advisor"], text=q["query"], relevant=frozenset(q["relevant"]))
        for q in golden["queries"]
    ]
    return documents, queries


def embed(model: TextEmbedding, texts: list[str]) -> list[Vector]:
    # Normalise explicitly: not every model returns unit vectors (e.g. nomic), and the dot
    # product of unit vectors is the cosine similarity used everywhere below.
    return [(vector / np.linalg.norm(vector)).astype(np.float32) for vector in model.embed(texts)]


def evaluate(variant: Variant, documents: list[Document], queries: list[Query]) -> Result:
    started = time.perf_counter()
    model = TextEmbedding(variant.model)
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    doc_vectors = dict(
        zip(
            [d.key for d in documents],
            embed(model, [variant.document_prefix + d.text for d in documents]),
            strict=True,
        )
    )
    ms_per_document = (time.perf_counter() - started) * 1000 / len(documents)

    query_vectors = embed(model, [variant.query_prefix + q.text for q in queries])

    hits, recalls3, recalls5, reciprocal_ranks = [], [], [], []
    relevant_scores: list[float] = []
    irrelevant_scores: list[float] = []
    top1_irrelevant: list[float] = []
    rankings: list[list[tuple[float, str]]] = []
    for query, q_vec in zip(queries, query_vectors, strict=True):
        candidates = [d.key for d in documents if d.advisor == query.advisor]
        scored = sorted(
            ((float(q_vec @ doc_vectors[key]), key) for key in candidates), reverse=True
        )
        ranking = [key for _, key in scored]
        rankings.append(scored)
        first_relevant = next(i for i, key in enumerate(ranking, start=1) if key in query.relevant)
        hits.append(1.0 if ranking[0] in query.relevant else 0.0)
        recalls3.append(len(query.relevant & set(ranking[:3])) / len(query.relevant))
        recalls5.append(len(query.relevant & set(ranking[:5])) / len(query.relevant))
        reciprocal_ranks.append(1 / first_relevant)
        for score, key in scored:
            (relevant_scores if key in query.relevant else irrelevant_scores).append(score)
        top1_irrelevant.append(next(s for s, key in scored if key not in query.relevant))

    canonical_q = embed(model, [variant.query_prefix + CANONICAL_QUERY])[0]
    canonical = {
        text: float(canonical_q @ vec)
        for text, vec in zip(CANONICAL_TEXTS, embed(model, CANONICAL_TEXTS), strict=True)
    }

    return Result(
        variant=variant,
        hit_at_1=float(np.mean(hits)),
        recall_at_3=float(np.mean(recalls3)),
        recall_at_5=float(np.mean(recalls5)),
        mrr=float(np.mean(reciprocal_ranks)),
        relevant_scores=relevant_scores,
        irrelevant_scores=irrelevant_scores,
        top1_irrelevant_scores=top1_irrelevant,
        rankings=rankings,
        load_seconds=load_seconds,
        ms_per_document=ms_per_document,
        canonical=canonical,
    )


def best_threshold(relevant: list[float], irrelevant: list[float]) -> tuple[float, float]:
    """Cut-off maximising F1 when treating 'score >= t' as 'relevant'."""
    best = (0.0, 0.0)
    for t in sorted(set(relevant + irrelevant)):
        tp = sum(s >= t for s in relevant)
        fp = sum(s >= t for s in irrelevant)
        fn = len(relevant) - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[1]:
            best = (t, f1)
    return best


def address_proof_ranks(queries: list[Query], rankings: list[list[str]]) -> str:
    for query, ranking in zip(queries, rankings, strict=True):
        if query.text == ADDRESS_PROOF_QUERY:
            return ", ".join(f"{key}={ranking.index(key) + 1}" for key in ADDRESS_PROOF_DOCS)
    return "-"


def rerank_metrics(model_name: str, documents: list[Document], queries: list[Query]) -> str:
    """Cross-encoder scoring every candidate document (no retrieval stage): a quality ceiling."""
    model = TextCrossEncoder(model_name)
    hits, recalls5, reciprocal_ranks = [], [], []
    rankings: list[list[str]] = []
    started = time.perf_counter()
    for query in queries:
        candidates = [d for d in documents if d.advisor == query.advisor]
        scores = list(model.rerank(query.text, [d.text for d in candidates]))
        order = sorted(zip(scores, candidates, strict=True), key=lambda pair: -pair[0])
        ranking = [d.key for _, d in order]
        rankings.append(ranking)
        first_relevant = next(i for i, key in enumerate(ranking, start=1) if key in query.relevant)
        hits.append(1.0 if ranking[0] in query.relevant else 0.0)
        recalls5.append(len(query.relevant & set(ranking[:5])) / len(query.relevant))
        reciprocal_ranks.append(1 / first_relevant)
    ms_per_query = (time.perf_counter() - started) * 1000 / len(queries)
    return (
        f"| rerank: {model_name} | {np.mean(hits):.2f} | {np.mean(recalls5):.2f} "
        f"| {np.mean(reciprocal_ranks):.2f} | {address_proof_ranks(queries, rankings)} "
        f"| {ms_per_query:.0f} ms/query |"
    )


def relative_cutoff_f1(result: Result, queries: list[Query], delta: float) -> float:
    """F1 when keeping only documents scoring within `delta` of the query's best score."""
    tp = fp = fn = 0
    for query, scored in zip(queries, result.rankings, strict=True):
        top = scored[0][0]
        kept = {key for score, key in scored if score >= top - delta}
        tp += len(kept & query.relevant)
        fp += len(kept - query.relevant)
        fn += len(query.relevant - kept)
    return 2 * tp / (2 * tp + fp + fn)


def lexical_overlap_count(documents: list[Document], queries: list[Query]) -> int:
    """How many queries share at least one word with a relevant document."""
    words = re.compile(r"[a-z0-9]+")
    texts = {d.key: set(words.findall(d.text.lower())) for d in documents}
    stop = {
        "a",
        "the",
        "for",
        "to",
        "of",
        "i",
        "if",
        "with",
        "from",
        "when",
        "how",
        "what",
        "where",
    }
    count = 0
    for q in queries:
        q_words = set(words.findall(q.text.lower())) - stop
        if any(q_words & texts[key] for key in q.relevant):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="larger models and rerankers")
    args = parser.parse_args()

    documents, queries = load_corpus()
    print(f"Corpus: {len(documents)} documents, {len(queries)} queries.")
    print(
        "Queries sharing a non-stop word with a relevant document: "
        f"{lexical_overlap_count(documents, queries)} of {len(queries)}.\n"
    )

    variants = VARIANTS + (EXTENDED_VARIANTS if args.extended else [])
    results = [evaluate(v, documents, queries) for v in variants]

    print("## Retrieval quality\n")
    print("| Model | hit@1 | recall@3 | recall@5 | MRR | 'address proof' ranks | ms/doc |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        ranks = address_proof_ranks(queries, [[k for _, k in scored] for scored in r.rankings])
        print(
            f"| {r.variant.label} | {r.hit_at_1:.2f} | {r.recall_at_3:.2f} | {r.recall_at_5:.2f} "
            f"| {r.mrr:.2f} | {ranks} | {r.ms_per_document:.1f} |"
        )

    print("\n## Score separation (cosine)\n")
    print(
        "| Model | relevant min / median | irrelevant median / max "
        "| best top-1 irrelevant, median | F1-optimal cut-off (F1) |"
    )
    print("|---|---|---|---|---|")
    for r in results:
        cut, f1 = best_threshold(r.relevant_scores, r.irrelevant_scores)
        print(
            f"| {r.variant.label} "
            f"| {min(r.relevant_scores):.3f} / {np.median(r.relevant_scores):.3f} "
            f"| {np.median(r.irrelevant_scores):.3f} / {max(r.irrelevant_scores):.3f} "
            f"| {np.median(r.top1_irrelevant_scores):.3f} | {cut:.3f} ({f1:.2f}) |"
        )

    print("\n## Relative cut-off: keep documents within delta of the best score (F1)\n")
    deltas = [0.02, 0.04, 0.06, 0.08, 0.10]
    print("| Model | " + " | ".join(f"δ={d}" for d in deltas) + " |")
    print("|---|" + "---|" * len(deltas))
    for r in results:
        cells = " | ".join(f"{relative_cutoff_f1(r, queries, d):.2f}" for d in deltas)
        print(f"| {r.variant.label} | {cells} |")

    chosen = results[0]
    print(f"\n## Per-query ranking: {chosen.variant.label}\n")
    print("| Query | first relevant rank | top-3 (score) |")
    print("|---|---|---|")
    for query, scored in zip(queries, chosen.rankings, strict=True):
        rank = next(i for i, (_, k) in enumerate(scored, start=1) if k in query.relevant)
        top3 = ", ".join(
            f"**{k}** ({s:.3f})" if k in query.relevant else f"{k} ({s:.3f})" for s, k in scored[:3]
        )
        print(f"| {query.text} | {rank} | {top3} |")

    if args.extended:
        print("\n## Rerankers over all candidates (quality ceiling)\n")
        print("| Model | hit@1 | recall@5 | MRR | 'address proof' ranks | latency |")
        print("|---|---|---|---|---|---|")
        for name in RERANKERS:
            print(rerank_metrics(name, documents, queries))

    print(f'\n## Canonical pairs: "{CANONICAL_QUERY}" vs …\n')
    print("| Model | " + " | ".join(CANONICAL_TEXTS) + " |")
    print("|---|" + "---|" * len(CANONICAL_TEXTS))
    for r in results:
        cells = " | ".join(f"{r.canonical[t]:.3f}" for t in CANONICAL_TEXTS)
        print(f"| {r.variant.label} | {cells} |")


if __name__ == "__main__":
    main()
