This is a solid project spec. Here's how to approach it in order:

---

## Phase 0 — Environment & Data (30 min)

**Set up first:**
- Python 3.11+, `uv` or `venv`
- PostgreSQL with `pgvector` extension (Docker is the fastest path)
- Get your embedding provider key — OpenAI `text-embedding-3-small` (1536-dim) is the default match for the schema above; `nomic-embed-text` via Ollama works if you want free/local

**Pick your corpus now** (don't defer this):
- GDPR full text — available as plain text/markdown on EUR-Lex
- Ethiopian Labour Proclamation — relevant to your context, publicly available
- LlamaIndex or FastAPI docs — if you want a dev-docs variant

Download it as `.txt` or `.md` before writing a line of code.

---

## Phase 1 — Ingestion Pipeline (build this raw, no LlamaIndex yet)

**Order:**
1. Write the SQL schema exactly as given — run it, confirm the indexes exist
2. Write a chunker: sliding window, ~500 tokens, 50-token overlap — use `tiktoken` for token counting, not character splits
3. Write an ingestor: read file → chunk → embed (batched) → insert `document` + `chunk` rows
4. Confirm rows are in the DB with vectors populated before moving on

**Don't touch FastAPI yet.** Scripts first.

---

## Phase 2 — Hybrid Retrieval (the core skill)

Write two separate retrieval functions and test them independently:

- `vector_search(query, k)` — embed query, run `ORDER BY embedding <=> $1 LIMIT k`
- `keyword_search(query, k)` — run `WHERE ts @@ plainto_tsquery('english', $1)`

Then write a merger (RRF — Reciprocal Rank Fusion is the standard approach). This becomes your `retrieve(query, k=20)` function.

**Test this before building generation.** Print the top 5 chunks for 3-4 known questions. If retrieval is wrong here, generation will never fix it.

---

## Phase 3 — Rerank + Generate

- Rerank: pass top 20 chunks to the LLM with a simple prompt — "rank these by relevance to the question, return IDs in order" — take top 5
- Generate: strict prompt — "answer only from the provided context, cite chunk IDs inline, return `{answer, citations}`"
- Handle the no-answer case explicitly in the prompt: "if the answer is not in the context, return `{answer: 'I don't know', citations: []}`"

---

## Phase 4 — FastAPI Wrapper

Two endpoints only:
- `POST /documents` — accepts text body + metadata, runs ingestion
- `POST /query` — accepts question string, returns `{answer, citations}`

Add this last. The pipeline logic should already be working as functions.

---

## Phase 5 — Eval Set (the portfolio differentiator)

Write 15-20 question/answer pairs where you know the correct source chunk. Measure:
- **Retrieval hit-rate**: is the correct chunk in the top 5? top 20?
- **Generation accuracy**: separate metric

Run this eval at two chunk sizes (e.g., 300 vs 500 tokens) and put the comparison in your README. This is what separates this project from a tutorial clone.

---

## Recommended File Structure

```
rag-service/
├── db/
│   └── schema.sql
├── ingest/
│   ├── chunker.py
│   ├── embedder.py
│   └── ingestor.py
├── retrieval/
│   ├── vector_search.py
│   ├── keyword_search.py
│   └── hybrid.py
├── generation/
│   ├── reranker.py
│   └── generator.py
├── api/
│   └── main.py          ← FastAPI, added last
├── eval/
│   ├── questions.json
│   └── eval_retrieval.py
└── README.md
```

---