# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hybrid-retrieval RAG service over a legal corpus (Ethiopian Labour Proclamation, in `data/`). Raw pipeline — no LlamaIndex/LangChain. Postgres + pgvector is the only store; everything else is plain functions wired together by `api/main.py`.

## Commands

Dependencies are managed with `uv` (see `uv.lock`); `requirements.txt` also present.

```bash
# Start the vector DB (Postgres 16 + pgvector)
docker run --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb \
  -p 5432:5432 -d pgvector/pgvector:pg16

# Apply schema (creates extension, tables, GIN index)
psql "$DATABASE_URL" -f db/schema.sql

# Run the API
uv run uvicorn api.main:app --reload

# Run retrieval eval (DB must already be populated)
uv run python eval/eval_retrieval.py
```

No test framework or linter is configured. The eval script (`eval/eval_retrieval.py`) is the de-facto correctness check — it prints per-question HIT/MISS for vector / keyword / hybrid retrieval and a top-k hit-rate summary. A "hit" = a retrieved chunk contains any of the question's expected keywords (`eval/questions.json`). It is **not** a pytest test; run it directly.

There is no CLI ingestion script (`main.py` was removed). Ingest a document either through `POST /documents` or by calling `ingest.ingestor.ingest_document(title, source, text)` directly.

## Required env (`.env`, see `.example.env`)

- `DATABASE_URL` — Postgres connection string (e.g. `postgresql://postgres:postgres@localhost:5432/ragdb`)
- `OPENROUTER_API_KEY` — used for embeddings, reranking, and generation
- `GEMINI_API_KEY` — read at import in `config.py` but not currently used by the pipeline; must still be set or `config.py` raises `KeyError` on import

## Architecture

Pipeline stages, each a standalone module that exposes plain functions. `api/main.py` is the only thing that composes them; build/test stages in isolation before touching the API.

```
ingest:    chunker → embedder → ingestor   (write path)
retrieval: vector_search + keyword_search → hybrid (RRF)   (read path)
generation: reranker → generator
api/main.py: POST /documents (ingest), POST /query (retrieve→rerank→generate), GET /health
```

Query flow (`/query`): `retrieve(q, k=20)` → `rerank(q, candidates, top_n=5)` → `generate_answer(q, top_chunks)` → `{answer, citations}`.

### Key design points

- **Single source of config.** `config.py` holds all models, dims, and tuning knobs (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`, `RETRIEVAL_K=20`, `RERANK_TOP_N=5`). Change behavior here, not in modules.

- **Embeddings are 4096-dim** (`qwen/qwen3-embedding-8b` via OpenRouter). The `chunk.embedding` column is `vector(4096)` and **has no ANN index** — pgvector HNSW/IVFFlat cap at 2000 dims, so vector search is an exact scan. Fine for <10k chunks. If you swap to a smaller-dim model, update `EMBEDDING_DIM` in `config.py` *and* the column type in `db/schema.sql` together. Despite the variable name, embeddings go through OpenRouter, not Gemini.

- **Hybrid = RRF.** `retrieval/hybrid.py` runs vector + keyword search independently, then merges by Reciprocal Rank Fusion (`RRF_K=60`), score = Σ 1/(RRF_K + rank + 1). Both lists are fetched at `k`, merged, then truncated to `k`.

- **Keyword search** uses a Postgres generated `tsvector` column (`ts`, English) with a GIN index, queried via `plainto_tsquery` + `ts_rank`. No app-side text processing.

- **Reranking and generation are LLM calls with primary+fallback models.** Both `reranker.py` and `generator.py` try `*_MODEL` then `*_MODEL_FALLBACK` on `RateLimitError`, and degrade gracefully (rerank → first `top_n` candidates; generate → `{"answer": "I don't know.", "citations": []}`). LLM JSON output is stripped of ```` ```json ```` fences before parsing.

- **Grounding contract.** `generator.py` prompts the LLM to answer ONLY from context chunks, cite chunk IDs, and return exactly `{"answer": "I don't know.", "citations": []}` when the answer isn't present. Chunks are passed as `[CHUNK <id>]\n<content>`; citations refer to those IDs.

- **DB access pattern.** `vector_search.py` and `ingestor.py` define their own `get_conn()` that calls `register_vector(conn)` (required to pass/receive Python lists as `vector`). `keyword_search.py` connects plainly (no vector traffic). IDs (document + chunk) are `uuid4` strings.
