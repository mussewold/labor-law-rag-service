from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingest.ingestor import ingest_document
from retrieval.hybrid import retrieve
from generation.reranker import rerank
from generation.generator import generate_answer
from config import RETRIEVAL_K, RERANK_TOP_N

app = FastAPI(title="Labour Law RAG API")

# --- Request/Response models ---

class IngestRequest(BaseModel):
    title: str
    source: str
    text: str

class IngestResponse(BaseModel):
    doc_id: str
    chunks: int

class QueryRequest(BaseModel):
    question: str
    retrieval_k: int = RETRIEVAL_K
    rerank_top_n: int = RERANK_TOP_N

class QueryResponse(BaseModel):
    answer: str
    citations: list[str]

# --- Endpoints ---

@app.post("/documents", response_model=IngestResponse)
def ingest(req: IngestRequest):
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



@app.get("/health")
def health():
    return {"status": "ok"}