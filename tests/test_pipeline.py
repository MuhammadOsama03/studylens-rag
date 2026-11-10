import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import index_documents, index_pdf


class PipelineTests(unittest.TestCase):
    @patch("app.store_embedded_chunks", return_value=2)
    @patch("app.embed_chunks")
    @patch("app.chunk_pages")
    @patch("app.load_pdf")
    def test_pdf_pipeline_preserves_stage_order(self, load, chunk, embed, store):
        load.return_value = [{"text": "A document", "source": "test.pdf", "page": 1}]
        chunk.return_value = [{"text": "A document", "source": "test.pdf", "page": 1, "chunk": 1}]
        embed.return_value = [{"text": "A document", "embedding": [0.1]}]
        self.assertEqual(index_pdf(Path("test.pdf")), 2)
        load.assert_called_once_with("test.pdf")
        chunk.assert_called_once_with(load.return_value)
        embed.assert_called_once_with(chunk.return_value)
        store.assert_called_once_with(embed.return_value)

    def test_empty_directory_does_not_index(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(index_documents(Path(folder)), (0, 0))


if __name__ == "__main__":
    unittest.main()
