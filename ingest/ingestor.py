import os
import uuid
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from ingest.chunker import chunk_text
from ingest.embedder import embed_texts

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def ingest_document(title: str, source: str, text: str) -> str:
    """Chunk, embed, and store a document. Returns the document ID."""
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)

    print(f"Chunked into {len(chunks)} chunks, embedding...")
    contents = [c.content for c in chunks]
    vectors = embed_texts(contents)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Insert document
            cur.execute(
                "INSERT INTO document (id, title, source) VALUES (%s, %s, %s)",
                (doc_id, title, source)
            )
            # Insert chunks
            for chunk, vector in zip(chunks, vectors):
                chunk_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO chunk (id, document_id, content, chunk_index, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (chunk_id, doc_id, chunk.content, chunk.chunk_index, chunk.token_count, vector)
                )
        conn.commit()

    print(f"Ingested doc_id={doc_id} with {len(chunks)} chunks.")
    return doc_id