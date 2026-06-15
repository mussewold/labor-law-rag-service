# Labour Law RAG Service

A from-scratch Retrieval-Augmented Generation (RAG) service over a legal corpus
(the Ethiopian Labour Proclamation). Built without LlamaIndex or LangChain — the
entire pipeline is plain Python functions backed by **PostgreSQL + pgvector**.

It combines **hybrid retrieval** (dense vector search + keyword full-text search,
merged with Reciprocal Rank Fusion), **LLM reranking**, and **grounded generation**
that answers strictly from retrieved context and cites the chunks it used.



![labor_law_rag_screenshot](readme_image/labor_law_rag_screenshot.png)





---

## Features

- **Hybrid retrieval** — dense embeddings (`pgvector`) + Postgres full-text search,
  fused with Reciprocal Rank Fusion (RRF).
- **LLM reranking** — top-K candidates reranked by an LLM, narrowed to the top-N.
- **Grounded answers with citations** — the generator answers only from context and
  returns `{"answer", "citations"}`; it replies `"I don't know."` when the answer
  isn't in the retrieved chunks.
- **Graceful fallbacks** — rerank and generation each have a primary + fallback model
  and degrade safely on rate limits or bad output.
- **Retrieval evaluation** — a hit-rate harness compares vector vs. keyword vs. hybrid
  retrieval over a labeled question set.

---

## Architecture

```
                    ┌──────────────────────────── api/main.py (FastAPI) ────────────────────────────┐
                    │                                                                                │
  POST /documents ──┼──> ingest_document() ──> chunker ──> embedder ──> Postgres (document, chunk)   │
                    │                                                                                │
  POST /query ──────┼──> retrieve() ──> rerank() ──> generate_answer() ──> {answer, citations}       │
                    │        │                                                                       │
                    └────────┼───────────────────────────────────────────────────────────────────────┘
                             │
                  ┌──────────┴──────────┐
                  │   hybrid retrieval  │
                  ├─────────────────────┤
                  │ vector_search  ─┐   │
                  │                 ├─► RRF merge ─► top-k
                  │ keyword_search ─┘   │
                  └─────────────────────┘
```

| Stage       | Module                      | Responsibility                                                |
|-------------|-----------------------------|---------------------------------------------------------------|
| Ingest      | `ingest/chunker.py`         | Token-based sliding-window chunking (`tiktoken`, `cl100k_base`)|
|             | `ingest/embedder.py`        | Batched embeddings via OpenRouter                             |
|             | `ingest/ingestor.py`        | Chunk → embed → insert `document` + `chunk` rows             |
| Retrieve    | `retrieval/vector_search.py`| Cosine similarity over `vector(4096)` (exact scan)            |
|             | `retrieval/keyword_search.py`| Postgres `tsvector` full-text search + `ts_rank`            |
|             | `retrieval/hybrid.py`       | Reciprocal Rank Fusion merge                                  |
| Generate    | `generation/reranker.py`    | LLM reranks candidates, keeps top-N                          |
|             | `generation/generator.py`   | Grounded answer + citations as JSON                          |
| API         | `api/main.py`               | `POST /documents`, `POST /query`, `GET /health`             |
| Eval        | `eval/eval_retrieval.py`    | Retrieval hit-rate (vector vs. keyword vs. hybrid)          |
| Config      | `config.py`                 | All models, dims, and tuning knobs in one place             |

---

## Requirements

- **Python 3.12+**
- **Docker** (for Postgres + pgvector), or a local Postgres 16 with the `vector` extension
- **[uv](https://github.com/astral-sh/uv)** (recommended) or `pip`
- API keys:
  - `OPENROUTER_API_KEY` — used for embeddings, reranking, and generation
  - `GEMINI_API_KEY` — currently read at startup but unused by the pipeline; must still be set

---

## Setup

### 1. Clone & install dependencies

```bash
# With uv (recommended)
uv sync

# Or with pip + venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example file and fill in your values:

```bash
cp .example.env .env
```

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb
OPENROUTER_API_KEY=sk-or-...
GEMINI_API_KEY=your_key_here
```

### 3. Start the database

```bash
docker run --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

### 4. Apply the schema

Creates the `vector` extension, the `document` and `chunk` tables, the generated
`tsvector` column, and the GIN full-text index:

```bash
psql "$DATABASE_URL" -f db/schema.sql
```

---

## Running the project

### Start the API

```bash
uv run uvicorn api.main:app --reload
```

The API is now at `http://127.0.0.1:8000` (interactive docs at `/docs`).

### Ingest a document

There is no standalone ingestion CLI — ingest through the API. To load the bundled
corpus (`data/labour_proclamation.md`):

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{title:"Ethiopian Labour Proclamation", source:"data/labour_proclamation.md", text:.}' data/labour_proclamation.md)"
```

Response:

```json
{ "doc_id": "…uuid…", "chunks": 42 }
```

> You can also call `ingest.ingestor.ingest_document(title, source, text)` directly
> from a Python shell if you prefer.

### Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maximum normal working hours per week?"}'
```

Response:

```json
{
  "answer": "Normal hours of work shall not exceed eight hours a day or forty-eight hours a week.",
  "citations": ["…chunk_id…"]
}
```

Optional overrides: `retrieval_k` (default 20) and `rerank_top_n` (default 5).

### Health check

```bash
curl http://127.0.0.1:8000/health   # -> {"status": "ok"}
```

---

## Evaluation

The eval harness measures **retrieval hit-rate** — whether the chunk containing the
expected answer appears in the top-k retrieved results — separately for vector,
keyword, and hybrid retrieval. Labeled questions live in `eval/questions.json`.

The database must already be populated (ingest the corpus first):

```bash
uv run python eval/eval_retrieval.py
```

Sample output:

```
============================================================
RETRIEVAL HIT-RATE (top-20)
  Vector only : 15/15 = 100%
  Keyword only: 6/15  = 40%
  Hybrid (RRF): 15/15 = 100%
============================================================
```

The takeaway: dense vector search dominates on paraphrased questions where keyword
search misses, while hybrid retrieval keeps the best of both. Adjust `CHUNK_SIZE` in
`config.py` and re-run to compare chunking strategies.

---

## Configuration

All tuning lives in `config.py`:

| Setting                | Default                      | Purpose                                  |
|------------------------|------------------------------|------------------------------------------|
| `EMBEDDING_MODEL`      | `qwen/qwen3-embedding-8b`    | Embedding model (via OpenRouter)         |
| `EMBEDDING_DIM`        | `4096`                       | Must match `vector(…)` in `schema.sql`   |
| `RERANK_MODEL`         | `qwen/qwen3.5-flash-02-23`   | Reranking LLM (+ fallback)               |
| `GENERATION_MODEL`     | `openai/gpt-4o-mini`         | Answer LLM (+ fallback)                  |
| `RETRIEVAL_K`          | `20`                         | Candidates fetched per retrieval         |
| `RERANK_TOP_N`         | `5`                          | Chunks kept after rerank → generation    |
| `CHUNK_SIZE`           | `500`                        | Tokens per chunk                         |
| `CHUNK_OVERLAP`        | `50`                         | Token overlap between chunks             |

> **Changing the embedding model?** Update `EMBEDDING_DIM` **and** the `embedding`
> column type in `db/schema.sql` together, then re-ingest. Note: `pgvector`'s HNSW/
> IVFFlat indexes cap at 2000 dims, so the 4096-dim column uses an exact scan (no ANN
> index) — fine for small/medium corpora (<10k chunks).

---

## Project structure

```
rag-1/
├── api/
│   └── main.py              # FastAPI app: /documents, /query, /health
├── ingest/
│   ├── chunker.py           # tiktoken sliding-window chunker
│   ├── embedder.py          # batched OpenRouter embeddings
│   └── ingestor.py          # chunk → embed → store
├── retrieval/
│   ├── vector_search.py     # pgvector cosine search
│   ├── keyword_search.py    # Postgres full-text search
│   └── hybrid.py            # Reciprocal Rank Fusion
├── generation/
│   ├── reranker.py          # LLM reranking
│   └── generator.py         # grounded answer + citations
├── eval/
│   ├── questions.json       # labeled eval questions
│   └── eval_retrieval.py    # hit-rate harness
├── db/
│   └── schema.sql           # tables, vector column, FTS index
├── data/
│   └── labour_proclamation.md
├── config.py                # all models & tuning knobs
└── README.md
```

---

## How it works

1. **Ingestion** — text is split into ~500-token chunks (50-token overlap) using
   `tiktoken`, each chunk is embedded in batches, and `document` + `chunk` rows are
   written to Postgres. The `ts` column (a generated `tsvector`) is populated
   automatically for keyword search.
2. **Retrieval** — the query is embedded for vector search and passed verbatim to
   keyword search; both return top-k, then RRF fuses the two ranked lists into one.
3. **Reranking** — an LLM scores the fused candidates by relevance and keeps the top-N.
4. **Generation** — the reranked chunks are formatted as `[CHUNK <id>]` blocks and the
   LLM answers strictly from them, returning JSON with the answer and cited chunk IDs,
   or `"I don't know."` when the answer isn't present.
