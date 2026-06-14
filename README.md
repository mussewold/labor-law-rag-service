# Step by step


## 1. Start pgvector on docker 
```
  docker run --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
  ```

