import unittest
from unittest.mock import patch

import retrieval


class RetrievalTests(unittest.TestCase):
    def test_rejects_invalid_top_k(self):
        for value in (0, retrieval.MAX_TOP_K + 1, 1.5):
            with self.assertRaises(ValueError):
                retrieval.semantic_search("question", limit=value)

    @patch("retrieval.get_collection")
    def test_empty_collection_skips_embedding(self, get_collection):
        get_collection.return_value.count.return_value = 0
        with patch("retrieval.embed_query") as embed_query:
            self.assertEqual(retrieval.semantic_search("question"), [])
            embed_query.assert_not_called()

    @patch("retrieval.semantic_search")
    def test_context_preserves_rank_and_source_metadata(self, semantic_search):
        semantic_search.return_value = [
            {"text": "Alpha", "metadata": {"source": "notes.pdf", "page": 2, "chunk": 4}, "distance": 0.12},
            {"text": "Beta", "metadata": {}, "distance": 0.33},
        ]
        context = retrieval.retrieve_context("What is alpha?", top_k=2)
        self.assertEqual(context[0]["rank"], 1)
        self.assertEqual(context[0]["source"], "notes.pdf")
        self.assertEqual(context[0]["page"], 2)
        self.assertIsNone(context[1]["source"])


if __name__ == "__main__":
    unittest.main()
