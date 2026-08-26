from config import client
from retrieval import retrieve_context


GENERATION_MODEL = "gemini-2.5-flash"


def _build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"Context {chunk['rank']}:\n{chunk['text']}"
        for chunk in chunks
    )


def answer_question(question: str, top_k: int = 5) -> dict:
    """Answer a question using only context retrieved from indexed documents."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question cannot be empty")

    chunks = retrieve_context(question, top_k=top_k)
    if not chunks:
        return {
            "answer": "I could not find relevant information in the indexed documents.",
            "sources": [],
        }

    context = _build_context(chunks)
    prompt = f"""You are StudyLens, a document question-answering assistant.
Answer the question using only the context below.
If the context does not contain enough information, say that the answer is not available in the indexed documents.
Do not add facts from outside the provided context.

Context:
{context}

Question: {question.strip()}
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    answer = response.text.strip() if response.text else "No answer was generated."
    return {"answer": answer, "sources": chunks}
