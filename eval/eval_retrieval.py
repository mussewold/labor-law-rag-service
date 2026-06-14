"""
Measures retrieval hit-rate separately from generation.
A hit = correct chunk appears in top-k retrieved results.
Run at two chunk sizes to compare.
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retrieval.hybrid import retrieve
from retrieval.vector_search import vector_search
from retrieval.keyword_search import keyword_search
from ingest.embedder import embed_query

def chunk_contains_keywords(chunk: dict, keywords: list[str]) -> bool:
    content = chunk["content"].lower()
    return any(kw.lower() in content for kw in keywords)

def evaluate(questions: list[dict], k: int = 20):
    vector_hits = 0
    keyword_hits = 0
    hybrid_hits = 0

    print(f"\n{'='*60}")
    print(f"Evaluating {len(questions)} questions, top-k={k}")
    print(f"{'='*60}\n")

    for q in questions:
        qid = q["id"]
        question = q["question"]
        keywords = q["keywords"]

        # Vector only
        vec = vector_search(embed_query(question), k=k)
        v_hit = any(chunk_contains_keywords(c, keywords) for c in vec)

        # Keyword only
        kw = keyword_search(question, k=k)
        k_hit = any(chunk_contains_keywords(c, keywords) for c in kw)

        # Hybrid
        hybrid = retrieve(question, k=k)
        h_hit = any(chunk_contains_keywords(c, keywords) for c in hybrid)

        vector_hits += v_hit
        keyword_hits += k_hit
        hybrid_hits += h_hit

        status = lambda hit: "HIT " if hit else "MISS"
        print(f"[{qid}] {question[:55]}")
        print(f"       Vector: {status(v_hit)} | Keyword: {status(k_hit)} | Hybrid: {status(h_hit)}")

    n = len(questions)
    print(f"\n{'='*60}")
    print(f"RETRIEVAL HIT-RATE (top-{k})")
    print(f"  Vector only : {vector_hits}/{n} = {vector_hits/n:.0%}")
    print(f"  Keyword only: {keyword_hits}/{n} = {keyword_hits/n:.0%}")
    print(f"  Hybrid (RRF): {hybrid_hits}/{n} = {hybrid_hits/n:.0%}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    with open("eval/questions.json") as f:
        questions = json.load(f)
    evaluate(questions, k=20)