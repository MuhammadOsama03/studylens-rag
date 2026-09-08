import unittest

from vector_store import _chunk_id, _validate_embedding


class VectorStoreValidationTests(unittest.TestCase):
    def test_embedding_validation_accepts_numeric_vectors(self):
        _validate_embedding([0.1, 2, -0.3])

    def test_embedding_validation_rejects_empty_or_non_numeric_vectors(self):
        with self.assertRaises(ValueError):
            _validate_embedding([])
        with self.assertRaises(ValueError):
            _validate_embedding([0.1, "bad"])

    def test_chunk_id_is_stable_and_changes_with_chunk_identity(self):
        base = {"source": "notes.pdf", "page": 2, "chunk": 1}
        self.assertEqual(_chunk_id(base), _chunk_id(dict(base)))
        self.assertNotEqual(_chunk_id(base), _chunk_id({**base, "chunk": 2}))


if __name__ == "__main__":
    unittest.main()
