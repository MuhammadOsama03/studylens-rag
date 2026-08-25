from google.genai import types

from config import client


EMBEDDING_MODEL = "gemini-embedding-001"
MAX_EMBEDDING_BATCH = 100


def _validate_chunks(chunks: list[dict]) -> None:
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise TypeError("each chunk must be a dictionary")
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError("each chunk must contain non-empty text")


def _embed_batch(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"].strip() for chunk in chunks]

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    if not response.embeddings or len(response.embeddings) != len(chunks):
        raise RuntimeError("Gemini returned an unexpected number of embeddings")

    return [
        {
            **chunk,
            "embedding": embedding.values,
            "embedding_model": EMBEDDING_MODEL,
        }
        for chunk, embedding in zip(chunks, response.embeddings)
    ]


def embed_chunks(
    chunks: list[dict],
    batch_size: int = MAX_EMBEDDING_BATCH,
) -> list[dict]:
    """Generate retrieval embeddings in batches while keeping chunk metadata."""
    _validate_chunks(chunks)

    if not chunks:
        return []
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    embedded_chunks = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embedded_chunks.extend(_embed_batch(batch))

    return embedded_chunks
