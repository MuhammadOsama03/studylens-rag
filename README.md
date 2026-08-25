# StudyLens RAG

StudyLens is a small RAG project I am building to understand how document-based AI systems work.

The idea is simple: load a PDF, extract its text, split it into useful chunks, create embeddings, store them locally, and later retrieve the most relevant parts for a question before sending that context to Gemini.

## Flow

PDF → text extraction → chunking → embeddings → ChromaDB → retrieval → Gemini → answer with source/page reference

## Tech Stack

- Python
- PyMuPDF
- Google Gemini API
- ChromaDB
- python-dotenv

## Current Progress

The document indexing side is now in place. StudyLens can extract PDF text with page metadata, split it into overlapping chunks, generate Gemini embeddings in batches, and store the embedded chunks in a local ChromaDB collection.

Next I am working on retrieval and question answering.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key
```

PDF files and the local ChromaDB data are kept out of Git.
