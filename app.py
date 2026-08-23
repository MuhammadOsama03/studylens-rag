from pathlib import Path

from pdf_loader import load_pdf


def main():
    pdf_files = list(Path("data").glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in the data folder.")
        print("Add a PDF to data/ and run the app again.")
        return

    file_path = pdf_files[0]
    pages = load_pdf(str(file_path))

    print(f"Loaded: {file_path.name}")
    print(f"Pages with extracted text: {len(pages)}\n")

    if pages:
        first_page = pages[0]
        preview = first_page["text"][:500]

        print(f"Source: {first_page['source']}")
        print(f"Page: {first_page['page']}")
        print("\nPreview:\n")
        print(preview)


if __name__ == "__main__":
    main()
