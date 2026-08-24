def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Split page text into overlapping chunks while keeping metadata."""
    if not isinstance(pages, list):
        raise TypeError("pages must be a list")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks = []

    for page in pages:
        if not isinstance(page, dict):
            raise TypeError("each page must be a dictionary")

        required_fields = {"text", "page", "source"}
        missing_fields = required_fields - page.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"page is missing required fields: {missing}")

        text = page["text"]
        if not isinstance(text, str):
            raise TypeError("page text must be a string")

        if not text.strip():
            continue

        start = 0
        chunk_number = 1

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                        "source": page["source"],
                        "chunk": chunk_number,
                    }
                )

            start += chunk_size - overlap
            chunk_number += 1

    return chunks
