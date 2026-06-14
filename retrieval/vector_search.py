import os
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]

def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def vector_search(query_embedding: list[float], k: int = 20) -> list[dict]:
    """Return top-k chunks by cosine similarity."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, c.content, c.chunk_index,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM chunk c
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, k)
            )
            rows = cur.fetchall()

    return [
        {"id": r[0], "document_id": r[1], "content": r[2], "chunk_index": r[3], "score": float(r[4])}
        for r in rows
    ]