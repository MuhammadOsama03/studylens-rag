from pathlib import Path

from chunker import chunk_pages
from pdf_loader import load_pdf


def main():
    pdf_files = list(Path("data").glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in the data folder.")
        print("Add a PDF to data/ and run the app again.")
        return

    file_path = pdf_files[0]
    pages = load_pdf(str(file_path))
    chunks = chunk_pages(pages)

    print(f"Loaded: {file_path.name}")
    print(f"Pages with extracted text: {len(pages)}")
    print(f"Chunks created: {len(chunks)}\n")

    for chunk in chunks[:3]:
        print(
            f"Source: {chunk['source']} | "
            f"Page: {chunk['page']} | "
            f"Chunk: {chunk['chunk']}"
        )
        print(chunk["text"][:300])
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
