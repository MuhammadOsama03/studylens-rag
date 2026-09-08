import tempfile
import unittest
from pathlib import Path

from pdf_loader import load_pdf


class PdfLoaderValidationTests(unittest.TestCase):
    def test_missing_pdf_raises_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "PDF not found"):
            load_pdf("missing-study-document.pdf")

    def test_non_pdf_file_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            path.write_text("Study notes", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be a PDF"):
                load_pdf(str(path))


if __name__ == "__main__":
    unittest.main()
