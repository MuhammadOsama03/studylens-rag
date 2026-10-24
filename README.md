# StudyLens RAG

StudyLens is a small RAG project I am building to understand how document-based AI systems work.

The idea is simple: load PDFs, extract their text, split it into useful chunks, create embeddings, store them locally, and retrieve the most relevant parts before sending grounded context to Gemini.

## Flow

PDFs → text extraction → chunking → embeddings → ChromaDB → semantic search → Top-K retrieval → Gemini → answer with source/page references

## Tech Stack

- Python
- PyMuPDF
- Google Gemini API
- ChromaDB
- python-dotenv

## Current Progress

The core RAG pipeline is now in place. StudyLens can index multiple PDFs, preserve page metadata, generate Gemini embeddings in batches, store chunks in a local ChromaDB collection, retrieve relevant context with semantic search, and generate grounded answers with source/page references.

The CLI also supports adjustable retrieval depth with `/topk N` so retrieval behavior can be tested without changing the code.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key
```

Add one or more PDFs to the `data/` folder and run:

```bash
python app.py
```

PDF files, API secrets, and local ChromaDB data are kept out of Git.

## Next Step

Run the complete pipeline with real documents and refine retrieval behavior based on the results.

## Quality checks

Run offline unit tests before indexing private documents:

```bash
python -m unittest discover -s tests -v
```

The tests mock Gemini for RAG and pipeline checks, so no API key is required for those test cases. GitHub Actions runs the same suite on pushes and pull requests.

### Real-document acceptance test

1. Put a non-sensitive, text-based PDF in `data/` with a known fact on a known page.
2. Configure `GEMINI_API_KEY` locally and run `python app.py`.
3. Ask a question answered by the PDF and confirm the answer and source/page reference.
4. Ask a question that the document cannot answer and check whether the assistant declines to invent facts.
5. Re-index after editing a PDF and verify obsolete chunks are no longer retrieved.

**Limitations:** The current pipeline needs text-extractable PDFs; scanned/image-only PDFs require OCR. Source citations identify retrieved pages, not independently verified claims. The real Gemini integration must be validated with your own key and representative documents.
