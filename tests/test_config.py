import os
import unittest
from unittest.mock import patch

import config


class ConfigTests(unittest.TestCase):
    def test_missing_key_is_checked_lazily(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                config.get_client()

    @patch("config.genai.Client")
    def test_configured_key_builds_client(self, client):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            self.assertIs(config.get_client(), client.return_value)
            client.assert_called_once_with(api_key="test-key")


if __name__ == "__main__":
    unittest.main()
