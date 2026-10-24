import unittest
from unittest.mock import patch, MagicMock

from rag import _build_context, _format_sources, answer_question


class RagTests(unittest.TestCase):
    def test_context_includes_page_references(self):
        context = _build_context([{"rank": 1, "source": "manual.pdf", "page": 2, "text": "Safety first."}])
        self.assertIn("manual.pdf", context)
        self.assertIn("page: 2", context)

    def test_sources_are_deduplicated(self):
        chunks = [
            {"source": "a.pdf", "page": 1},
            {"source": "a.pdf", "page": 1},
            {"source": "a.pdf", "page": 2},
        ]
        self.assertEqual(len(_format_sources(chunks)), 2)

    @patch("rag.retrieve_context", return_value=[])
    def test_no_retrieval_does_not_call_model(self, retrieve):
        with patch("rag.get_client") as client:
            result = answer_question("Unknown question")
            self.assertEqual(result["sources"], [])
            client.assert_not_called()

    @patch("rag.retrieve_context")
    @patch("rag.get_client")
    def test_grounded_answer_uses_retrieved_context(self, client, retrieve):
        retrieve.return_value = [{"rank": 1, "source": "manual.pdf", "page": 3, "text": "The deadline is Friday."}]
        client.return_value.models.generate_content.return_value = MagicMock(text="Friday.")
        result = answer_question("When is the deadline?")
        self.assertEqual(result["answer"], "Friday.")
        self.assertEqual(result["sources"], [{"source": "manual.pdf", "page": 3}])
        prompt = client.return_value.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("The deadline is Friday.", prompt)
        self.assertIn("Do not add facts", prompt)


if __name__ == "__main__":
    unittest.main()
