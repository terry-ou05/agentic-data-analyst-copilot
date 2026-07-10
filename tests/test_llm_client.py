import sys
from types import SimpleNamespace

import pytest

from src.llm import client as llm_client


def _install_fake_openai(monkeypatch, *, response=None, error=None) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            if error is not None:
                raise error
            return response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))


@pytest.fixture(autouse=True)
def isolated_llm_environment(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "_load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid")


def _response_with_content(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_normal_content_is_returned(monkeypatch) -> None:
    _install_fake_openai(
        monkeypatch,
        response=_response_with_content("  valid response  "),
    )

    result = llm_client.generate_chat_completion([{"role": "user", "content": "test"}])

    assert result.success is True
    assert result.content == "valid response"
    assert result.error == ""


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_or_non_text_content_fails(monkeypatch, content) -> None:
    _install_fake_openai(monkeypatch, response=_response_with_content(content))

    result = llm_client.generate_chat_completion([{"role": "user", "content": "test"}])

    assert result.success is False
    assert result.content == ""
    assert "invalid or empty response" in result.error


def test_empty_choices_fails_without_index_error(monkeypatch) -> None:
    _install_fake_openai(monkeypatch, response=SimpleNamespace(choices=[]))

    result = llm_client.generate_chat_completion([{"role": "user", "content": "test"}])

    assert result.success is False
    assert "invalid or empty response" in result.error


def test_missing_choices_fails(monkeypatch) -> None:
    _install_fake_openai(monkeypatch, response=SimpleNamespace())

    result = llm_client.generate_chat_completion([{"role": "user", "content": "test"}])

    assert result.success is False
    assert "invalid or empty response" in result.error


def test_missing_message_fails(monkeypatch) -> None:
    _install_fake_openai(
        monkeypatch,
        response=SimpleNamespace(choices=[SimpleNamespace()]),
    )

    result = llm_client.generate_chat_completion([{"role": "user", "content": "test"}])

    assert result.success is False
    assert "invalid or empty response" in result.error


def test_missing_response_fails(monkeypatch) -> None:
    _install_fake_openai(monkeypatch, response=None)

    result = llm_client.generate_chat_completion([{"role": "user", "content": "test"}])

    assert result.success is False
    assert "invalid or empty response" in result.error


@pytest.mark.parametrize("error", [TimeoutError("do-not-expose"), RuntimeError("do-not-expose")])
def test_api_exception_is_sanitized(monkeypatch, error) -> None:
    _install_fake_openai(monkeypatch, error=error)

    result = llm_client.generate_chat_completion([{"role": "user", "content": "test"}])

    assert result.success is False
    assert error.__class__.__name__ in result.error
    assert "do-not-expose" not in result.error
