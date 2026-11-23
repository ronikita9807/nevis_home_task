# Nevis Search API

[![CI](https://github.com/ronikita9807/nevis_home_task/actions/workflows/ci.yml/badge.svg)](https://github.com/ronikita9807/nevis_home_task/actions/workflows/ci.yml)

A search API for wealth advisors over their clients and the clients' documents:

- **clients** by email, name or description: `NevisWealth` → `john.doe@neviswealth.com`,
  and by the meaning of the description: `lawyer` → *"Partner at a law firm…"*;
- **documents** by meaning and by domain synonyms: `address proof` → documents containing
  `utility bill`, `council tax bill`, `tenancy agreement`.

Python 3.13 · FastAPI · PostgreSQL 17 with `pgvector` and `pg_trgm` · local embedding model
(`bge-small-en-v1.5` via fastembed/ONNX, no API keys) · Docker Compose.

---

## Quickstart

Requirements: Docker with Compose v2.

```bash
docker compose up --build -d --wait        # db, migrations, api  (or: make up)
docker compose exec api python scripts/seed.py   # demo data       (or: make seed)
```

- API: <http://localhost:8000>, Swagger UI: <http://localhost:8000/docs>
  (click **Authorize** and set `X-Advisor-Id`).
- The seed creates two demo advisors and prints their ids:

| Advisor | `X-Advisor-Id` | Data |
|---|---|---|
| Alice | `a1a1a1a1-0000-4000-8000-000000000001` | 6 clients, 19 documents |
| Bob | `b2b2b2b2-0000-4000-8000-000000000002` | 2 clients, 3 documents |

John Doe (`john.doe@neviswealth.com`) is a client of **both** advisors, with different
documents: try the same search as Alice and as Bob.

```bash
curl -s -G localhost:8000/search --data-urlencode "q=address proof" \
  -H "X-Advisor-Id: a1a1a1a1-0000-4000-8000-000000000001"
```

Ports are bound to `127.0.0.1`: API on `8000`, Postgres on `5433` (not `5432`, which is often
taken by a local Postgres). Override with `API_PORT` / `DB_PORT`.

---

## API

| Method | Path | Description | Success | Errors |
|---|---|---|---|---|
| `POST` | `/clients` | Create a client | 201 + `Location` | 401, 409, 422 |
| `POST` | `/clients/{id}/documents` | Add a document (indexed immediately) | 201 + `Location` | 401, 404, 422 |
| `GET` | `/search?q=` | Search clients and documents | 200 | 401, 422 |
| `GET` | `/clients/{id}` | Get a client *(extension)* | 200 | 401, 404 |
| `GET` | `/documents/{id}` | Get a document *(extension)* | 200 | 401, 404 |
| `GET` | `/documents/{id}/summary` | 2–3 sentence summary *(optional feature)* | 200 | 401, 404 |
| `GET` | `/health`, `/ready` | Liveness / readiness (database) | 200 | 503 |

Request and response bodies follow the assignment's OpenAPI spec; everything else is
additive. Full contract: Swagger UI, or [`docs/openapi.json`](docs/openapi.json) (kept in
sync by a test).

**Headers**

- `X-Advisor-Id` (required): the advisor on whose behalf the call is made. All data is
  scoped to it; another advisor's client or document is reported as `404`.
- `X-API-Key`: shared key of the trusted caller; checked only when the server sets `API_KEY`
  (not set in `docker compose`).

**`GET /search` parameters**: `q` (1–256 characters), optional `type=client|document`,
`limit` per kind (1–100, default 20).

**Document summary** (`GET /documents/{id}/summary`) returns
`{"document_id": "…", "summary": "…", "method": "llm" | "extractive"}`, cached after the first call.

- `extractive` (default): the 3 sentences closest to the document's mean embedding, in
  order. Local, deterministic, nothing leaves the service.
- `llm`: written by Claude (`claude-haiku-5-5`, low effort) when the server has
  `ANTHROPIC_API_KEY` (`ANTHROPIC_API_KEY=sk-ant-… docker compose up -d`). **This sends the
  document's text, i.e. client data, to Anthropic** — an operator decision, hence opt-in.
  The document is passed as delimited, untrusted data (instructions inside it are not
  followed). On timeout (10 s), API error, refusal or empty output the endpoint falls back to
  `extractive` instead of failing.

**Errors** are [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) Problem Details:

```json
{
  "type": "about:blank",
  "title": "Conflict",
  "status": 409,
  "detail": "You already have a client with this email",
  "instance": "/clients",
  "code": "duplicate_client_email",
  "request_id": "3539e08e6ce44538ae4bd5af7186dabc"
}
```

---

## Example searches

Responses from the seeded demo data as Alice (ids shortened, long fields trimmed).
The response is an array: **client hits first, then document hits**, each block ordered by
its own `score` (scores are comparable only within a block).

**Client by company domain**: `GET /search?q=NevisWealth`

```json
[
  {
    "type": "client",
    "score": 0.6,
    "matched_fields": ["email"],
    "matched_by": ["keyword"],
    "client": {
      "id": "08e4b15e-…",
      "first_name": "John",
      "last_name": "Doe",
      "email": "john.doe@neviswealth.com",
      "description": "Senior software engineer, 41, married with two children. …",
      "social_links": ["https://www.linkedin.com/in/john-doe-example"]
    }
  }
]
```

**Documents by synonym**: `GET /search?q=address proof&type=document&limit=3`

```json
[
  {
    "type": "document",
    "score": 0.047619,
    "matched_by": ["semantic", "keyword"],
    "snippet": "Agreement between landlord Mr Peter Hughes and tenant Ms Sophie Müller for the property at",
    "document": {"id": "ffd9ef6b-…", "client_id": "1c50f55f-…", "title": "Assured shorthold tenancy agreement", "created_at": "2026-11-23T10:17:51Z"},
    "client": {"id": "1c50f55f-…", "first_name": "Sophie", "last_name": "Müller"}
  },
  {
    "type": "document",
    "score": 0.032787,
    "matched_by": ["keyword"],
    "snippet": "London Borough of Camden council tax bill for 1 April 2024 to 31 March 2025. …",
    "document": {"id": "8d776958-…", "client_id": "99a1c02d-…", "title": "Council tax bill 2024/25", "created_at": "2026-11-23T10:17:50Z"},
    "client": {"id": "99a1c02d-…", "first_name": "Emily", "last_name": "Carter"}
  },
  {
    "type": "document",
    "score": 0.032258,
    "matched_by": ["keyword"],
    "snippet": "British Gas energy statement for the period 1 February to 29 February 2024. …",
    "document": {"id": "bb48f9bc-…", "client_id": "08e4b15e-…", "title": "Utility bill — British Gas statement, March 2024", "created_at": "2026-11-23T10:17:49Z"},
    "client": {"id": "08e4b15e-…", "first_name": "John", "last_name": "Doe"}
  }
]
```

**Documents by meaning, no shared words**: `GET /search?q=borrowing to buy a house&type=document&limit=1`

```json
[
  {
    "type": "document",
    "score": 0.016393,
    "matched_by": ["semantic"],
    "snippet": "Nationwide Building Society offer of a loan of £650,000 secured on 7 Harbour View, Brighton BN2 5TT. …",
    "document": {"id": "9c66cd51-…", "client_id": "4a483c0b-…", "title": "Mortgage offer — Nationwide", "created_at": "2026-11-23T10:17:50Z"},
    "client": {"id": "4a483c0b-…", "first_name": "Raj", "last_name": "Patel"}
  }
]
```

**Typo**: `GET /search?q=Jonh` → John Doe (`score: 0.2`, `matched_fields: ["email", "first_name"]`).

**Client by meaning of the description**: `GET /search?q=lawyer&type=client` → Liam O'Brien,
*"Partner at a law firm…"* (`score: 0.287`, `matched_by: ["semantic"]`); `doctor` → Raj Patel
(*"Consultant surgeon…"*).

**Isolation**: the same `q=NevisWealth` as Bob returns only *Bob's* John Doe record.

---

## How search works

### Clients

1. **SQL selects candidates** within the advisor's data: every query word must match some
   field (`email`, `first_name`, `last_name`, `description`) — case-insensitive `ILIKE` with
   `%`, `_`, `\` escaped — or be a typo of a word there (`pg_trgm` `<%` operator, words of
   4+ letters). GIN trigram indexes serve both.
2. **Python confirms and ranks** (pure functions, unit-tested): trigram similarity is only a
   pre-filter; a typo is confirmed by edit distance (≤1 edit for short words, ≤2 for longer),
   so `neviswealth` does not match the word `wealth`. Tiers: exact email or full name `1.0` >
   prefix `0.8` > all words found `0.6` > typo `0.5 × similarity`.
3. **By meaning of the description**: each description is embedded when the client is
   created; the query embedding finds the nearest descriptions. Measured on the demo data,
   name-like queries reach at most 0.45 similarity while meaning queries put the right client
   first at 0.50–0.65 with close runners-up, so a client counts only above **0.48** and
   within **0.03** of the best one (at most 3). Such matches score `0.5 × similarity` —
   below any word match — and carry `matched_by: ["semantic"]`.

### Documents

```
POST /clients/{id}/documents
  └─ ownership check ─ chunk (~200 words, sentence-aware, overlap) ─ embed (bge-small, CPU)
       └─ one short transaction: document (+ English tsvector) and chunk vectors (pgvector)

GET /search?q=
  ├─ semantic channel: query embedding → nearest chunks (HNSW, advisor filter,
  │    iterative scan) → best chunk per document → top 5 above a noise floor
  ├─ keyword channel: Postgres full-text search, query expanded with a domain thesaurus
  │    ("address proof" → utility bill, council tax bill, bank statement, …)
  └─ weighted Reciprocal Rank Fusion (keyword 2×, semantic 1×) → snippet + owning client
```

Why two channels and a thesaurus: [`docs/embedding-experiment.md`](docs/embedding-experiment.md)
compares 8 local embedding models and 3 rerankers on a golden set. The key finding: **no
local model knows that a utility bill is a proof of address** — for `address proof` the
utility bill ranked 8th–16th of 17 documents, even with "Utility bill" in its title. That is
domain knowledge, so it lives in a small, explainable thesaurus of document types.

### Quality

Golden set: 19 queries over the demo corpus, written to share **no words** with the
documents they should find (`make eval`):

| Setup | recall@5 | MRR |
|---|---|---|
| embeddings only (`bge-small-en-v1.5`) | 0.89 | 0.81 |
| full pipeline (semantic + thesaurus + weighted RRF) | **0.92** | **0.84** |

`address proof` puts the tenancy agreement, the council tax bill and the utility bill in the
top 3. Remaining misses are paraphrases outside the thesaurus, e.g. *money for when I stop
working* → pension statement.

---

## Architecture

Modules by domain, each with a framework-free core and an `infra/` layer (ports & adapters):

```
app/
  clients/     domain, command, DTO, ports, service, api, infra/ (ORM, mappers, repository)
  documents/   + chunking, Embedder port, fastembed adapter
  search/      read side: thesaurus, ranking (pure), SQL queries, service, api
  shared/      settings, database, errors (RFC 9457), security, logging, health
  containers.py  dependency injection: the only place ports meet implementations
migrations/    Alembic (pgvector, pg_trgm, tables, indexes)
```

- **Write side**: command → service → domain model → repository, with the transaction
  boundary in the service (unit of work). **Read side** (`search`): query services return
  DTOs directly and may read across modules' tables.
- Services depend on `Protocol` ports; tests swap the embedding model for a deterministic
  fake, so API tests are fast and independent of model weights.
- **Advisor isolation is enforced by the database**, not only by queries: every table has
  `advisor_id`; composite foreign keys (`documents → clients`, `chunks → documents`) make a
  document's owner equal to its client's owner; email is unique *per advisor*.

---

## Decisions & trade-offs

| Decision | Why | Cost |
|---|---|---|
| PostgreSQL + `pgvector` + `pg_trgm`, no separate search engine | one store for data, vectors, full-text and fuzzy search; transactional; enough for millions of chunks | no distributed scaling |
| Local `bge-small-en-v1.5` (ONNX, no torch) | runs with no keys/network; within noise of models 10–18× larger on the golden set; 517 MB image with the model baked in | lower quality than hosted embeddings |
| Domain thesaurus in the keyword channel | the domain fact *utility bill = proof of address* is unknown to every local model tested | a curated list to maintain |
| Weighted RRF (keyword 2×) | ranks, not incomparable raw scores; keyword hits are precise, semantic top hits are noisy | weights are a judgement, validated on the golden set |
| No relevance threshold, only a noise floor (0.5) | relevant/irrelevant similarities overlap; the floor removes noise without losing recall (measured) | broad queries may show loosely related documents |
| Synchronous indexing on `POST` | the document is searchable on 201; simple | ~0.2 s per KB of text on a laptop: 0.2–1 s for a typical 1–5 KB document, ~20 s for the 100 KB maximum (worker timeout set to 120 s); a queue in production |
| Embedding computed **outside** the DB transaction | no pooled connection is held during inference | ownership re-checked by the FK on insert |
| Response is the spec's array, client block then document block | the contract is kept; blocks avoid ranking clients against documents | see open questions |
| Email unique per advisor; another advisor's data → 404 | a global constraint would leak, via 409, that someone else has this client | — |

## Assumptions & open questions

In a real project I would settle these with the product owner before building. Here each is
an explicit assumption, cheap to change:

- **Language** — clients are UK/US, so English only (English model and stemming). Supporting
  e.g. Chinese means a multilingual or hosted model and a `simple` full-text config: one
  adapter, a migration and a re-index (`embedding_model` is stored per chunk).
- **Who calls the API** — assumed internal, behind a gateway/BFF that authenticates the advisor
  and passes `X-Advisor-Id`; the shared `X-API-Key` protects it when exposed. Front-end checks
  alone would not be security: a user can call the API directly.
- **Client ownership** — a client record belongs to exactly one advisor; if several advisors
  work with the same person, each has their own record and documents.
- **Response shape** — a grouped response (`{clients: [], documents: []}`) would suit a UI
  better; the spec asks for an array, so blocks + `type=` give the same behaviour without
  breaking the contract.
- **Documents** — plain text up to 100 KB; no file upload or OCR.
- **Third-party AI** — by default nothing leaves the service: embeddings and summaries are
  computed locally. LLM summaries are opt-in (`ANTHROPIC_API_KEY`) because they send client
  data to a third party.

## Production next steps

- Per-advisor authentication (SSO/OAuth → JWT with the advisor id) and Postgres Row-Level
  Security as a second line of defence; audit log of access to client data.
- Asynchronous indexing (queue + indexing status) for large documents; file upload with text
  extraction.
- Hosted embedding model and/or a cross-encoder reranker on the top results; thesaurus owned by
  compliance, or expanded offline with an LLM and reviewed.
- Pagination for search; rate limiting; metrics and tracing.

---

## Development

Requirements: [uv](https://docs.astral.sh/uv/), Docker (tests start Postgres with testcontainers).

```bash
make install     # dependencies
make check       # ruff, pyright (strict), mypy (strict), pytest — exactly what CI runs
make eval        # search quality on the golden set (recall@5, MRR)
make experiment  # embedding model comparison (ARGS=--extended for large models)
make openapi     # regenerate docs/openapi.json
make up / seed / logs / down
```

**Tests** (132) target behaviour that can break: the assignment's cases, isolation between
advisors, error contract, races (concurrent duplicate emails), LIKE escaping, chunking limits,
ranking rules, and search quality with the real model. They run against a real Postgres with
`pgvector`, migrated with Alembic.

Run the API outside Docker: `cp .env.example .env`, `docker compose up -d db`,
`uv run alembic upgrade head`, `uv run uvicorn --factory app.main:create_app --reload`.
