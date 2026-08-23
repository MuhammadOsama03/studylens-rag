# StudyLens RAG

StudyLens is a small RAG project I am building to understand how document-based AI systems work.

The idea is simple: load a PDF, extract its text, split it into useful chunks, store embeddings, retrieve the most relevant parts for a question, and then use Gemini to answer from that context.

## Planned Flow

PDF → text extraction → chunking → embeddings → vector search → Gemini → answer with source/page reference

## Tech Stack

- Python
- PyMuPDF
- Google Gemini API
- ChromaDB
- python-dotenv

## Current Progress

The repository is initialized. The first step is PDF text extraction while keeping the page number and source file with the extracted text.
