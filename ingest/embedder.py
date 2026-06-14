import time
from openai import OpenAI, RateLimitError
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

BATCH_SIZE = 50

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches via OpenRouter."""
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = texts[i:i + BATCH_SIZE]
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            all_embeddings.extend([e.embedding for e in response.data])
        except RateLimitError:
            print("  Rate limited, waiting 30s...")
            time.sleep(30)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            all_embeddings.extend([e.embedding for e in response.data])

    return all_embeddings

def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding