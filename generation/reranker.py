import json
from openai import OpenAI, RateLimitError
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, RERANK_MODEL, RERANK_MODEL_FALLBACK, RERANK_MAX_TOKENS

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    if not chunks:
        return []

    chunk_list = "\n\n".join(
        f"[{i}] {c['content'][:300]}" for i, c in enumerate(chunks)
    )

    prompt = f"""You are a relevance reranker. Given a question and a list of text chunks, return the indices of the most relevant chunks in order of relevance.

Question: {query}

Chunks:
{chunk_list}

Respond ONLY with a JSON array of indices, most relevant first. Example: [2, 0, 4, 1, 3]
Return at most {top_n} indices."""

    for model in [RERANK_MODEL, RERANK_MODEL_FALLBACK]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=RERANK_MAX_TOKENS
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            indices = json.loads(text)
            return [chunks[i] for i in indices if i < len(chunks)][:top_n]
        except RateLimitError:
            print(f"  Rate limited on {model}, trying fallback...")
            continue
        except (json.JSONDecodeError, IndexError):
            return chunks[:top_n]

    return chunks[:top_n]
