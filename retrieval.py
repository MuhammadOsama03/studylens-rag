from google.genai import types

from config import client
from embeddings import EMBEDDING_MODEL
from vector_store import get_collection


DEFAULT_TOP_K = 5


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


def semantic_search(question: str, limit: int = DEFAULT_TOP_K) -> list[dict]:
    """Search the local StudyLens collection for semantically similar chunks."""
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    collection = get_collection()
    collection_size = collection.count()
    if collection_size == 0:
        return []

    query_embedding = embed_query(question)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, collection_size),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {"text": text, "metadata": metadata or {}, "distance": distance}
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def retrieve_context(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the Top-K document chunks to use as context for an answer."""
    results = semantic_search(question, limit=top_k)

    return [
        {
            "rank": rank,
            "text": result["text"],
            "source": result["metadata"].get("source"),
            "page": result["metadata"].get("page"),
            "chunk": result["metadata"].get("chunk"),
            "distance": result["distance"],
        }
        for rank, result in enumerate(results, start=1)
    ]
