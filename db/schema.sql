CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document (
  id          TEXT PRIMARY KEY,
  title       TEXT NOT NULL,
  source      TEXT NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunk (
  id           TEXT PRIMARY KEY,
  document_id  TEXT REFERENCES document(id),
  content      TEXT NOT NULL,
  chunk_index  INT  NOT NULL,
  token_count  INT  NOT NULL,
  embedding    vector(1536)
);

CREATE INDEX ON chunk USING hnsw (embedding vector_cosine_ops);

ALTER TABLE chunk ADD COLUMN ts tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX ON chunk USING gin (ts);