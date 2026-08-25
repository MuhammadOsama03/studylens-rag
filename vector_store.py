from pathlib import Path

try:
    import chromadb
except ImportError as exc:
    raise ImportError(
        "ChromaDB is required for local vector storage. Install it with: pip install chromadb"
    ) from exc


DEFAULT_DB_PATH = Path("chroma_db")
DEFAULT_COLLECTION = "studylens_documents"


def get_collection(
    db_path: str | Path = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Open the local persistent ChromaDB collection used by StudyLens."""
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("collection_name cannot be empty")

    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(name=collection_name)


def store_embedded_chunks(
    embedded_chunks: list[dict],
    db_path: str | Path = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Persist embedded chunks and their metadata in local ChromaDB storage."""
    if not isinstance(embedded_chunks, list):
        raise TypeError("embedded_chunks must be a list")

    if not embedded_chunks:
        return 0

    collection = get_collection(db_path=db_path, collection_name=collection_name)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in embedded_chunks:
        if not isinstance(chunk, dict):
            raise TypeError("each embedded chunk must be a dictionary")

        text = chunk.get("text")
        embedding = chunk.get("embedding")
        source = chunk.get("source")
        page = chunk.get("page")
        chunk_number = chunk.get("chunk")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("each embedded chunk must contain non-empty text")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("each embedded chunk must contain an embedding")
        if source is None or page is None or chunk_number is None:
            raise ValueError("source, page, and chunk metadata are required")

        chunk_id = f"{source}:p{page}:c{chunk_number}"
        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in {"text", "embedding"} and value is not None
        }

        ids.append(chunk_id)
        documents.append(text)
        embeddings.append(embedding)
        metadatas.append(metadata)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)
