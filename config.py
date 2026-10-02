import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

def get_client() -> genai.Client:
    """Create the Gemini client only when an AI operation is requested."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your local .env file before running StudyLens."
        )
    return genai.Client(api_key=api_key)
