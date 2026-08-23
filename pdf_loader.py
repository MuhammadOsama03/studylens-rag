from pathlib import Path

import fitz


def load_pdf(file_path: str) -> list[dict]:
    """Extract text from a PDF while keeping page and source information."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    pages = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": path.name,
                }
            )

    return pages
