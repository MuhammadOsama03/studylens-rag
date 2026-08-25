from pathlib import Path

from chunker import chunk_pages
from embeddings import embed_chunks
from pdf_loader import load_pdf
from vector_store import store_embedded_chunks


def index_pdf(file_path: Path) -> int:
    """Load, chunk, embed, and store a PDF in the local vector database."""
    pages = load_pdf(str(file_path))
    chunks = chunk_pages(pages)

    if not chunks:
        return 0

    embedded_chunks = embed_chunks(chunks)
    return store_embedded_chunks(embedded_chunks)


def main():
    pdf_files = list(Path("data").glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in the data folder.")
        print("Add a PDF to data/ and run the app again.")
        return

    file_path = pdf_files[0]
    stored_chunks = index_pdf(file_path)

    print(f"Indexed: {file_path.name}")
    print(f"Chunks stored: {stored_chunks}")


if __name__ == "__main__":
    main()
