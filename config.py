import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models — primary + fallback
RERANK_MODEL = "qwen/qwen3.5-flash-02-23"
RERANK_MODEL_FALLBACK = "meta-llama/llama-3.2-3b-instruct:free"

GENERATION_MODEL = "openai/gpt-4o-mini"
GENERATION_MODEL_FALLBACK = "meta-llama/llama-3.3-70b-instruct:free"

# Gemini embedding
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_DIM = 4096


# Retrieval
RETRIEVAL_K = 20
RERANK_TOP_N = 5

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50