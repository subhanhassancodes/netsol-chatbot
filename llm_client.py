"""
llm_client.py

Single shared Gemini client + retry wrapper, used by chatbot.py (RAG /
Text2SQL / routing) and csv_analysis.py (CSV question answering) so there's
exactly one client configured and exactly one place that logs token usage
for every raw call made to Gemini (see token_tracker.py).
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

import token_tracker

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_NAME = "gemini-2.5-flash-lite"

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_with_retry(prompt: str, retries: int = 4, base_delay: float = 2.0) -> str:
    """Calls Gemini, retrying on transient errors like 503 UNAVAILABLE
    (temporary server overload — not a quota issue). Exponential backoff:
    2s, 4s, 8s, 16s between tries. Every successful call is logged to
    token_tracker so per-request and running token/cost totals stay
    accurate regardless of which code path triggered the call."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            usage = response.usage_metadata
            if usage is not None:
                token_tracker.log_usage(
                    MODEL_NAME,
                    usage.prompt_token_count or 0,
                    usage.candidates_token_count or 0,
                )
            return response.text
        except ServerError as e:
            last_error = e
            if attempt < retries:
                time.sleep(base_delay * (2 ** attempt))
    raise last_error

