from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ingest.ingestor import ingest_document
from retrieval.hybrid import retrieve
from generation.reranker import rerank
from generation.generator import generate_answer
from config import RETRIEVAL_K, RERANK_TOP_N

app = FastAPI(title="Labour Law RAG API")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/")
def index():
    return FileResponse(FRONTEND)

# --- Request/Response models ---

class IngestRequest(BaseModel):
    title: str
    source: str
    text: str

class IngestResponse(BaseModel):
    doc_id: str
    chunks: int

class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int

class QueryRequest(BaseModel):
    question: str
    retrieval_k: int = RETRIEVAL_K
    rerank_top_n: int = RERANK_TOP_N

class QueryResponse(BaseModel):
    answer: str
    citations: list[str]

class TraceChunk(BaseModel):
    id: str
    chunk_index: int
    score: float | None = None
    rrf_score: float | None = None
    preview: str

class VerboseResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved: list[TraceChunk]
    reranked: list[TraceChunk]
    timings_ms: dict[str, float]


def _trace(chunks: list[dict]) -> list[TraceChunk]:
    return [
        TraceChunk(
            id=c["id"],
            chunk_index=c["chunk_index"],
            score=c.get("score"),
            rrf_score=c.get("rrf_score"),
            preview=c["content"][:240],
        )
        for c in chunks
    ]

# --- Endpoints ---

@app.post("/documents", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not req.title.strip() or not req.source.strip() or not req.text.strip():
        raise HTTPException(status_code=400, detail="title, source, and text are all required.")
    try:
        doc_id = ingest_document(req.title, req.source, req.text)
        # count chunks for response
        import psycopg, os
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunk WHERE document_id = %s", (doc_id,))
                count = cur.fetchone()[0]
        return IngestResponse(doc_id=doc_id, chunks=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    """Remove a document and its chunks. Recovery path for a mistaken ingest."""
    import psycopg, os
    try:
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM document WHERE id = %s", (doc_id,))
                if cur.fetchone() is None:
                    raise HTTPException(status_code=404, detail=f"No document with id {doc_id}.")
                # FK has no ON DELETE CASCADE; remove chunks first, then the document.
                cur.execute("DELETE FROM chunk WHERE document_id = %s", (doc_id,))
                deleted = cur.rowcount
                cur.execute("DELETE FROM document WHERE id = %s", (doc_id,))
            conn.commit()
        return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        candidates = retrieve(req.question, k=req.retrieval_k)
        top_chunks = rerank(req.question, candidates, top_n=req.rerank_top_n)
        result = generate_answer(req.question, top_chunks)
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query/verbose", response_model=VerboseResponse)
def query_verbose(req: QueryRequest):
    import time
    try:
        t0 = time.perf_counter()
        candidates = retrieve(req.question, k=req.retrieval_k)
        t1 = time.perf_counter()
        top_chunks = rerank(req.question, candidates, top_n=req.rerank_top_n)
        t2 = time.perf_counter()
        result = generate_answer(req.question, top_chunks)
        t3 = time.perf_counter()
        return VerboseResponse(
            answer=result["answer"],
            citations=result["citations"],
            retrieved=_trace(candidates),
            reranked=_trace(top_chunks),
            timings_ms={
                "retrieve": round((t1 - t0) * 1000, 1),
                "rerank": round((t2 - t1) * 1000, 1),
                "generate": round((t3 - t2) * 1000, 1),
                "total": round((t3 - t0) * 1000, 1),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}