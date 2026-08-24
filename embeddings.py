from google.genai import types

from config import client


EMBEDDING_MODEL = "gemini-embedding-001"


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Generate retrieval embeddings while keeping chunk metadata."""
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")

    if not chunks:
        return []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise TypeError("each chunk must be a dictionary")
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError("each chunk must contain non-empty text")

    texts = [chunk["text"] for chunk in chunks]

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    if not response.embeddings or len(response.embeddings) != len(chunks):
        raise RuntimeError("Gemini returned an unexpected number of embeddings")

    embedded_chunks = []

    for chunk, embedding in zip(chunks, response.embeddings):
        embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding.values,
                "embedding_model": EMBEDDING_MODEL,
            }
        )

    return embedded_chunks
