from pathlib import Path
import hashlib

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

    db_path = Path(db_path)
    if db_path.exists() and not db_path.is_dir():
        raise ValueError("db_path must point to a directory")

    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(name=collection_name.strip())


def _validate_embedding(embedding: list) -> None:
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("each embedded chunk must contain an embedding")
    if not all(isinstance(value, (int, float)) for value in embedding):
        raise ValueError("embedding values must be numeric")


def _chunk_id(chunk: dict) -> str:
    """Create a stable ID so re-indexing updates the same chunk position."""
    identity = f"{chunk['source']}|{chunk['page']}|{chunk['chunk']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def store_embedded_chunks(
    embedded_chunks: list[dict],
    db_path: str | Path = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Persist embedded chunks and replace older chunks for the same source PDFs."""
    if not isinstance(embedded_chunks, list):
        raise TypeError("embedded_chunks must be a list")

    if not embedded_chunks:
        return 0

    ids = []
    documents = []
    embeddings = []
    metadatas = []
    sources = set()

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
        _validate_embedding(embedding)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source metadata must be a non-empty string")
        if not isinstance(page, int) or page < 1:
            raise ValueError("page metadata must be a positive integer")
        if not isinstance(chunk_number, int) or chunk_number < 1:
            raise ValueError("chunk metadata must be a positive integer")

        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in {"text", "embedding"} and value is not None
        }

        sources.add(source.strip())
        ids.append(_chunk_id(chunk))
        documents.append(text.strip())
        embeddings.append(embedding)
        metadatas.append(metadata)

    collection = get_collection(db_path=db_path, collection_name=collection_name)

    # Remove the previous version of each source first. Without this, re-indexing a
    # shortened or re-chunked PDF can leave obsolete chunks behind in retrieval.
    for source in sources:
        collection.delete(where={"source": source})

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)
