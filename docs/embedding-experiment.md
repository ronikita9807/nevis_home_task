# Embedding model experiment

Reproduce: `uv run python scripts/embedding_experiment.py --extended`
(the extended run downloads several GB of model weights; drop the flag for the small models only).

## Setup

- Corpus: `data/demo_corpus.json` — 22 realistic wealth-management documents (KYC, tax,
  pensions, estate planning…) for 8 clients of two advisors. The same corpus seeds the demo.
- Queries: `data/golden_queries.json` — 19 queries with the documents a human expects.
  Queries deliberately share no words with their relevant documents (only 2 of 19 share any
  non-stop word), so they measure semantic matching, not keyword overlap.
- Candidates are the documents of the querying advisor only (as in the real API).
- Documents are embedded whole (`title + content`): each fits the 512-token window.
- Metrics: hit@1, recall@3/5, MRR; separation of cosine scores between relevant and
  irrelevant documents; latency on a laptop CPU.

## Findings

1. **Small local models are close to large ones on general paraphrases.** `bge-small-en-v1.5`
   (67 MB, 384d) reaches MRR 0.81 vs 0.86 for `bge-large-en-v1.5` (1.2 GB, 1024d), which is
   ~4–5× slower. With 19 queries one query is ±0.05, so most differences are within noise.
2. **The assignment's headline case is not solved by any local model.** For
   *"address proof"* the utility bill ranks 8th–16th of 17 documents and the council tax bill
   7th–17th — even with "Utility bill" in the document's title; passports and driving licences
   rank higher. Models know "a document used for verification", but not the domain fact
   "a utility bill is the standard UK proof of address". The best cross-encoder reranker
   (`bge-reranker-v2-m3`) lifts them to 3rd–4th at ~2 s per query and 570 MB — not a fit for
   synchronous search on CPU.
3. **An absolute similarity cut-off is unreliable.** Relevant and irrelevant scores overlap
   (bge-small: relevant median 0.60, irrelevant max 0.65; best F1 for any cut-off ≈ 0.5).
   A cut-off relative to the best score is no better (F1 ≤ 0.65).
4. **bge query prefix** (`Represent this sentence for searching relevant passages: `) slightly
   improves hit@1 and MRR (0.74 / 0.81 vs 0.68 / 0.79). fastembed does not add it — the adapter must.
5. `intfloat/multilingual-e5-small` is not available in fastembed (relevant if multilingual
   search is ever required: `multilingual-e5-large`, 2.2 GB, is).

## Decisions

- **Model:** `BAAI/bge-small-en-v1.5` via fastembed, with the query prefix. Quality is close to
  models 10–18× larger; small image, fast CPU inference. Swappable via settings + one adapter.
- **Domain thesaurus for the lexical channel.** Search expands the query with a curated list of
  document-type synonyms (e.g. *proof of address* → utility bill, council tax bill, bank
  statement, tenancy agreement) before full-text search; RRF then merges it with the semantic
  channel. This is what makes *"address proof"* return documents containing *"utility bill"*:
  deterministic, explainable to compliance, instant, testable. The thesaurus describes
  document categories, not golden queries — most golden queries are not covered by it and keep
  measuring the embeddings honestly.
- **No absolute similarity threshold.** The semantic channel proposes a bounded number of best
  documents (top-N, configurable); keyword/thesaurus matches are always included; RRF orders
  the union.
- **Weighted RRF (added in stage 6):** keyword matches weigh 2×, semantic 1×. With equal
  weights the semantic channel's noisy top hit (a driving licence for *"address proof"*)
  tied with exact thesaurus matches; queries that share no words with any document are
  unaffected (their keyword channel is empty).
- **Not now (production next steps):** a hosted embedding model and/or a reranker on the top
  results; thesaurus maintained by compliance/ops or expanded offline with an LLM.

## End-to-end result (stage 6)

The whole pipeline (API → semantic + keyword/thesaurus channels → weighted RRF) on the same
golden set, `make eval` (`tests/search/test_golden_set.py`):

| Setup | recall@5 | MRR | "address proof" top 3 |
|---|---|---|---|
| bge-small alone (this experiment) | 0.89 | 0.81 | tenancy agreement only |
| full pipeline | **0.92** | **0.84** | tenancy agreement, council tax bill, utility bill |

Remaining misses are paraphrases outside the thesaurus and beyond the small model, e.g.
*"money for when I stop working"* → pension statement (not in the top 5): candidates for a
hosted embedding model or a reranker on the top results.

## Raw results

Reproduce: `make experiment ARGS=--extended`.

Corpus: 22 documents, 19 queries.
Queries sharing a non-stop word with a relevant document: 2 of 19.

### Retrieval quality

| Model | hit@1 | recall@3 | recall@5 | MRR | 'address proof' ranks | ms/doc |
|---|---|---|---|---|---|---|
| bge-small-en-v1.5 + query prefix | 0.74 | 0.78 | 0.89 | 0.81 | john_utility_bill=12, emily_council_tax=17, sophie_tenancy=3 | 32.9 |
| bge-small-en-v1.5, no prefix | 0.68 | 0.83 | 0.91 | 0.79 | john_utility_bill=14, emily_council_tax=13, sophie_tenancy=3 | 31.7 |
| all-MiniLM-L6-v2 | 0.63 | 0.81 | 0.89 | 0.77 | john_utility_bill=13, emily_council_tax=15, sophie_tenancy=1 | 11.1 |
| arctic-embed-s + query prefix | 0.74 | 0.75 | 0.83 | 0.82 | john_utility_bill=8, emily_council_tax=10, sophie_tenancy=1 | 19.9 |
| bge-base-en-v1.5 + query prefix (768d) | 0.79 | 0.78 | 0.83 | 0.84 | john_utility_bill=9, emily_council_tax=7, sophie_tenancy=3 | 99.0 |
| bge-large-en-v1.5 + query prefix (1024d) | 0.79 | 0.86 | 0.89 | 0.86 | john_utility_bill=10, emily_council_tax=9, sophie_tenancy=1 | 142.4 |
| mxbai-embed-large-v1 + query prefix (1024d) | 0.74 | 0.86 | 0.89 | 0.84 | john_utility_bill=13, emily_council_tax=8, sophie_tenancy=1 | 139.1 |
| nomic-embed-text-v1.5 (768d) | 0.68 | 0.81 | 0.94 | 0.81 | john_utility_bill=16, emily_council_tax=7, sophie_tenancy=2 | 62.3 |

### Score separation (cosine)

| Model | relevant min / median | irrelevant median / max | best top-1 irrelevant, median | F1-optimal cut-off (F1) |
|---|---|---|---|---|
| bge-small-en-v1.5 + query prefix | 0.454 / 0.599 | 0.503 / 0.651 | 0.596 | 0.591 (0.51) |
| bge-small-en-v1.5, no prefix | 0.480 / 0.643 | 0.537 / 0.697 | 0.611 | 0.640 (0.53) |
| all-MiniLM-L6-v2 | 0.014 / 0.311 | 0.121 / 0.465 | 0.257 | 0.276 (0.54) |
| arctic-embed-s + query prefix | 0.460 / 0.570 | 0.458 / 0.609 | 0.531 | 0.552 (0.62) |
| bge-base-en-v1.5 + query prefix (768d) | 0.389 / 0.537 | 0.414 / 0.576 | 0.488 | 0.536 (0.57) |
| bge-large-en-v1.5 + query prefix (1024d) | 0.413 / 0.550 | 0.446 / 0.599 | 0.531 | 0.546 (0.53) |
| mxbai-embed-large-v1 + query prefix (1024d) | 0.434 / 0.578 | 0.458 / 0.629 | 0.558 | 0.558 (0.53) |
| nomic-embed-text-v1.5 (768d) | 0.466 / 0.581 | 0.501 / 0.630 | 0.576 | 0.609 (0.59) |

### Relative cut-off: keep documents within delta of the best score (F1)

| Model | δ=0.02 | δ=0.04 | δ=0.06 | δ=0.08 | δ=0.1 |
|---|---|---|---|---|---|
| bge-small-en-v1.5 + query prefix | 0.62 | 0.50 | 0.45 | 0.35 | 0.29 |
| bge-small-en-v1.5, no prefix | 0.57 | 0.50 | 0.42 | 0.34 | 0.27 |
| all-MiniLM-L6-v2 | 0.57 | 0.60 | 0.51 | 0.49 | 0.44 |
| arctic-embed-s + query prefix | 0.53 | 0.48 | 0.39 | 0.30 | 0.29 |
| bge-base-en-v1.5 + query prefix (768d) | 0.62 | 0.50 | 0.45 | 0.38 | 0.30 |
| bge-large-en-v1.5 + query prefix (1024d) | 0.59 | 0.47 | 0.40 | 0.33 | 0.27 |
| mxbai-embed-large-v1 + query prefix (1024d) | 0.65 | 0.50 | 0.43 | 0.37 | 0.30 |
| nomic-embed-text-v1.5 (768d) | 0.59 | 0.58 | 0.48 | 0.40 | 0.29 |

### Per-query ranking: bge-small-en-v1.5 + query prefix

| Query | first relevant rank | top-3 (score) |
|---|---|---|
| address proof | 3 | raj_driving_licence (0.596), john_passport (0.552), **sophie_tenancy** (0.540) |
| identity documents | 1 | **john_passport** (0.641), **raj_driving_licence** (0.591), jose_inheritance (0.523) |
| money for when I stop working | 8 | emily_suitability (0.611), raj_p60 (0.584), jose_bank_statement (0.576) |
| how much the client earns | 1 | **raj_p60** (0.595), emily_risk_profile (0.590), emily_suitability (0.553) |
| where the money came from | 1 | **jose_inheritance** (0.581), john_utility_bill (0.543), john_annual_review (0.511) |
| succession planning for heirs | 1 | **liam_will** (0.647), john_annual_review (0.625), **liam_life_policy** (0.599) |
| ethical investing | 4 | sophie_isa (0.630), emily_suitability (0.626), emily_risk_profile (0.619) |
| appetite for volatility | 2 | emily_suitability (0.622), **emily_risk_profile** (0.616), john_annual_review (0.597) |
| borrowing to buy a house | 1 | **raj_mortgage_offer** (0.673), emily_suitability (0.641), john_annual_review (0.612) |
| sheltered savings allowance | 4 | john_pension (0.651), john_annual_review (0.615), jose_bank_statement (0.612) |
| rental contract | 1 | **sophie_tenancy** (0.676), raj_mortgage_offer (0.602), emily_business_sale (0.595) |
| someone to manage finances if I become incapacitated | 1 | **liam_lpa** (0.617), emily_risk_profile (0.565), emily_suitability (0.552) |
| justification for the advice given | 1 | **emily_suitability** (0.626), emily_risk_profile (0.611), john_annual_review (0.577) |
| what was discussed with the client last summer | 1 | **john_annual_review** (0.585), emily_risk_profile (0.562), emily_business_sale (0.558) |
| insurance protection for family | 1 | **liam_life_policy** (0.636), liam_lpa (0.567), john_pension (0.560) |
| cash from selling a company | 1 | **emily_business_sale** (0.625), jose_inheritance (0.596), emily_suitability (0.582) |
| combining old retirement pots | 1 | **john_bob_workplace_pension** (0.591), hannah_sa302 (0.460), hannah_school_fees (0.423) |
| freelance earnings assessment | 1 | **hannah_sa302** (0.597), john_bob_workplace_pension (0.519), hannah_school_fees (0.487) |
| children's education costs | 1 | **hannah_school_fees** (0.675), hannah_sa302 (0.596), john_bob_workplace_pension (0.514) |

### Rerankers over all candidates (quality ceiling)

| Model | hit@1 | recall@5 | MRR | 'address proof' ranks | latency |
|---|---|---|---|---|---|
| rerank: Xenova/ms-marco-MiniLM-L-12-v2 | 0.58 | 0.88 | 0.70 | john_utility_bill=5, emily_council_tax=10, sophie_tenancy=3 | 330 ms/query |
| rerank: jinaai/jina-reranker-v1-turbo-en | 0.53 | 0.75 | 0.66 | john_utility_bill=17, emily_council_tax=10, sophie_tenancy=5 | 217 ms/query |
| rerank: BAAI/bge-reranker-v2-m3-int8 | 0.74 | 1.00 | 0.83 | john_utility_bill=3, emily_council_tax=4, sophie_tenancy=1 | 2064 ms/query |

### Canonical pairs: "address proof" vs …

| Model | utility bill | council tax bill | bank statement | stock portfolio |
|---|---|---|---|---|
| bge-small-en-v1.5 + query prefix | 0.521 | 0.459 | 0.549 | 0.490 |
| bge-small-en-v1.5, no prefix | 0.595 | 0.515 | 0.627 | 0.535 |
| all-MiniLM-L6-v2 | 0.092 | 0.046 | 0.372 | 0.140 |
| arctic-embed-s + query prefix | 0.446 | 0.448 | 0.464 | 0.412 |
| bge-base-en-v1.5 + query prefix (768d) | 0.423 | 0.352 | 0.478 | 0.305 |
| bge-large-en-v1.5 + query prefix (1024d) | 0.468 | 0.436 | 0.551 | 0.396 |
| mxbai-embed-large-v1 + query prefix (1024d) | 0.457 | 0.421 | 0.546 | 0.384 |
| nomic-embed-text-v1.5 (768d) | 0.373 | 0.372 | 0.473 | 0.374 |
