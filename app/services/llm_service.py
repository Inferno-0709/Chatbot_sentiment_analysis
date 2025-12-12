# app/services/llm_service.py

import os
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env (recommended)
load_dotenv()

# Get API key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")

# Initialize client
client = genai.Client(api_key=API_KEY)

# ----- Prompt Template -----
BASE_TEMPLATE = """
You are a helpful AI assistant.

The conversation so far is:
{history}

User: {user_message}

Reply naturally as the assistant:
"""


def build_prompt(history: str, user_message: str) -> str:
    """Injects history + user message into template."""
    return BASE_TEMPLATE.format(history=history.strip(), user_message=user_message.strip())


def generate_reply(history: str, user_message: str, model: str = "gemini-2.5-flash-lite") -> str:
    """
    Generate a bot reply using Gemini.
    
    Args:
        history (str): Previous chat transcript in plain text form.
        user_message (str): Latest user message.
        model (str): Gemini model name.

    Returns:
        str: Bot's reply.
    """
    prompt = build_prompt(history, user_message)

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    # Safely extract text
    return getattr(response, "text", "") or ""
