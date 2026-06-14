import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]

def keyword_search(query: str, k: int = 20) -> list[dict]:
    """Return top-k chunks by full-text keyword match."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, c.content, c.chunk_index,
                       ts_rank(c.ts, plainto_tsquery('english', %s)) AS score
                FROM chunk c
                WHERE c.ts @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, k)
            )
            rows = cur.fetchall()

    return [
        {"id": r[0], "document_id": r[1], "content": r[2], "chunk_index": r[3], "score": float(r[4])}
        for r in rows
    ]