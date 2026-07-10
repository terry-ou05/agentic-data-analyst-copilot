import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


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

    load_dotenv(dotenv_path=DOTENV_PATH, override=True)


def get_llm_config() -> dict:
    """Return non-sensitive LLM configuration for UI and diagnostics."""
    _load_dotenv_if_available()

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    return {
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
        "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        "api_key_configured": bool(api_key),
    }


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

    config = get_llm_config()
    model = config["model"]
    base_url = config["base_url"]

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
            error=(
                f"LLM request failed ({exc.__class__.__name__}). "
                "Please try again."
            ),
        )

    try:
        if response is None:
            raise ValueError("response is missing")

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("response choices are missing or empty")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("response message is missing")

        content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise TypeError("response content is not text")

        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("response content is empty")
    except Exception as exc:
        return LLMResult(
            success=False,
            error=(
                "LLM returned an invalid or empty response "
                f"({exc.__class__.__name__}). Please try again."
            ),
        )

    return LLMResult(success=True, content=normalized_content)
