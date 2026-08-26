import unittest

from chunker import chunk_pages


class ChunkerTests(unittest.TestCase):
    def test_chunk_pages_preserves_metadata(self):
        pages = [
            {
                "text": "A" * 900,
                "page": 2,
                "source": "sample.pdf",
            }
        ]

        chunks = chunk_pages(pages, chunk_size=500, overlap=100)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page"], 2)
        self.assertEqual(chunks[0]["source"], "sample.pdf")
        self.assertEqual(chunks[0]["chunk"], 1)
        self.assertTrue(chunks[0]["text"])

    def test_chunk_pages_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_pages(
                [{"text": "hello world", "page": 1, "source": "sample.pdf"}],
                chunk_size=100,
                overlap=100,
            )


if __name__ == "__main__":
    unittest.main()
