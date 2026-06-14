import tiktoken
from dataclasses import dataclass
from config import CHUNK_SIZE, CHUNK_OVERLAP

@dataclass
class Chunk:
    content: str
    chunk_index: int
    token_count: int

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        content = enc.decode(chunk_tokens)
        chunks.append(Chunk(
            content=content,
            chunk_index=index,
            token_count=len(chunk_tokens)
        ))
        index += 1
        start += chunk_size - overlap

    return chunks
