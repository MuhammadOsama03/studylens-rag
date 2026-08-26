from google.genai import types

from config import client
from embeddings import EMBEDDING_MODEL
from vector_store import get_collection


def embed_query(question: str) -> list[float]:
    """Create a query embedding for semantic retrieval."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question cannot be empty")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question.strip(),
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("Gemini did not return a query embedding")

    return response.embeddings[0].values


def semantic_search(question: str, limit: int = 5) -> list[dict]:
    """Search the local StudyLens collection for semantically similar chunks."""
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_query(question)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {"text": text, "metadata": metadata or {}, "distance": distance}
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]
