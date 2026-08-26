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
                for source in result["sources"]:
                    print(f"- {source['source']} — page {source['page']}")
                print()
        except Exception as exc:
            print(f"Error: {exc}\n")


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

    ask_questions()


if __name__ == "__main__":
    main()
