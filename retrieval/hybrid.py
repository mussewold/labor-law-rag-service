from retrieval.vector_search import vector_search
from retrieval.keyword_search import keyword_search
from ingest.embedder import embed_query

RRF_K = 60

def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict]
) -> list[dict]:
    """Merge two ranked lists using RRF."""
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunks[cid] = chunk

    for rank, chunk in enumerate(keyword_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunks[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"rrf_score": score, **chunks[cid]} for cid, score in ranked]

def retrieve(query: str, k: int = 20) -> list[dict]:
    """Full hybrid retrieval: vector + keyword → RRF merge → top k."""
    query_vec = embed_query(query)
    v_results = vector_search(query_vec, k=k)
    kw_results = keyword_search(query, k=k)
    merged = reciprocal_rank_fusion(v_results, kw_results)
    return merged[:k]