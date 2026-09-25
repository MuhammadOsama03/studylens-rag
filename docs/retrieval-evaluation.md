# Retrieval evaluation guide

Retrieval quality should be measured before changing chunk size, embeddings, ranking, or the default top-k value.

## Prepare a small benchmark

Create questions from documents you are allowed to use. For each question, record the expected source file and page, plus whether the available documents can answer it. Do not commit private documents or API keys.

## Run an evaluation

1. Index a fixed document set from a clean vector store.
2. Run every question with the same configuration.
3. Record the top retrieved source/page pairs and scores.
4. Check whether the expected evidence appears in the top-k results.
5. Confirm unanswerable questions are not answered as facts.
6. Repeat after a retrieval change and compare results.

## Suggested measures

- Recall at k for the expected page.
- Mean reciprocal rank of the first relevant chunk.
- Citation correctness in the final answer.
- Abstention rate for unanswerable questions.
- Indexing and query latency.

Record the dataset version, embedding model, chunk settings, and top-k value with every result so experiments remain reproducible.
