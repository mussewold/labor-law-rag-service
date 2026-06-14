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
  embedding    vector(4096)
  -- NOTE: no ANN index — pgvector HNSW/IVFFlat both cap at 2000 dims
  -- exact scan is fine for small-medium corpora (<10k chunks)
  -- options to fix later: reduce dims via PCA, or use a 768/1536-dim model
);

ALTER TABLE chunk ADD COLUMN ts tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX ON chunk USING gin (ts);
