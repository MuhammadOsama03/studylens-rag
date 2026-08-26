from pathlib import Path

from chunker import chunk_pages
from embeddings import embed_chunks
from pdf_loader import load_pdf
from rag import answer_question
from vector_store import store_embedded_chunks


def index_pdf(file_path: Path) -> int:
    """Load, chunk, embed, and store a PDF in the local vector database."""
    pages = load_pdf(str(file_path))
    chunks = chunk_pages(pages)

    if not chunks:
        return 0

    embedded_chunks = embed_chunks(chunks)
    return store_embedded_chunks(embedded_chunks)


def index_documents(data_dir: Path = Path("data")) -> tuple[int, int]:
    """Index every PDF in the data folder and return document/chunk totals."""
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        return 0, 0

    total_chunks = 0
    for file_path in pdf_files:
        stored_chunks = index_pdf(file_path)
        total_chunks += stored_chunks
        print(f"Indexed: {file_path.name} ({stored_chunks} chunks)")

    return len(pdf_files), total_chunks


def ask_questions() -> None:
    print("\nStudyLens is ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = answer_question(question)
            print(f"\nStudyLens: {result['answer']}\n")

            if result["sources"]:
                print("Sources:")
                seen_sources = set()
                for source in result["sources"]:
                    source_key = (source.get("source"), source.get("page"))
                    if source_key in seen_sources:
                        continue
                    seen_sources.add(source_key)
                    print(f"- {source['source']} — page {source['page']}")
                print()
        except Exception as exc:
            print(f"Error: {exc}\n")


def main():
    document_count, chunk_count = index_documents()

    if document_count == 0:
        print("No PDF found in the data folder.")
        print("Add one or more PDFs to data/ and run the app again.")
        return

    print(f"\nDocuments indexed: {document_count}")
    print(f"Total chunks stored: {chunk_count}")

    ask_questions()


if __name__ == "__main__":
    main()
