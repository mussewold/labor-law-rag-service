import json
from openai import OpenAI, RateLimitError
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, GENERATION_MODEL, GENERATION_MODEL_FALLBACK, GENERATION_MAX_TOKENS

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

def generate_answer(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"answer": "I don't know.", "citations": []}

    context = "\n\n".join(
        f"[CHUNK {c['id']}]\n{c['content']}" for c in chunks
    )
    print(f"Context:")
    print(context)

    prompt = f"""You are a precise HR legal/ HR regulatory assistant. Answer the question using ONLY the provided context chunks.

Rules:
- If the answer is in the context, answer clearly and cite the chunk IDs you used.
- If the answer is NOT in the context, respond exactly: {{"answer": "I don't know.", "citations": []}}
- Never hallucinate or use outside knowledge.
- Always respond with valid JSON in this exact format:
  {{"answer": "<your answer>", "citations": ["<chunk_id>", ...]}}

Context:
{context}

Question: {query}"""

    for model in [GENERATION_MODEL, GENERATION_MODEL_FALLBACK]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=GENERATION_MAX_TOKENS
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except RateLimitError:
            print(f"  Rate limited on {model}, trying fallback...")
            continue
        except json.JSONDecodeError:
            return {"answer": response.choices[0].message.content, "citations": []}

    return {"answer": "I don't know.", "citations": []}
