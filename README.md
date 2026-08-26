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
