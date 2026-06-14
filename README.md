# Step by step


## 1. Start pgvector on docker 
```
  docker run --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
  ```



## Folder Structure

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