import os
from dataclasses import dataclass


@dataclass
class LLMResult:
    success: bool
    content: str = ""
    error: str = ""


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def generate_chat_completion(messages: list[dict]) -> LLMResult:
    """Call an OpenAI-compatible chat completion API."""
    _load_dotenv_if_available()

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return LLMResult(
            success=False,
            error=(
                "LLM_API_KEY is missing. Create a .env file from .env.example "
                "and set LLM_API_KEY before generating an analysis plan."
            ),
        )

    model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    try:
        from openai import OpenAI
    except ImportError:
        return LLMResult(
            success=False,
            error=(
                "The openai package is not installed. Run "
                "`pip install -r requirements.txt` and try again."
            ),
        )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    except Exception as exc:
        return LLMResult(
            success=False,
            error=f"LLM request failed: {exc.__class__.__name__}: {exc}",
        )

    content = response.choices[0].message.content or ""
    return LLMResult(success=True, content=content.strip())
