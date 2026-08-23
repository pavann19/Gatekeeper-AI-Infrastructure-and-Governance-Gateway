"""
Tests for core.llm_providers (Phase 5, Real LLM Gateway — provider
abstraction only, per the roadmap's own item breakdown). Every test mocks
requests.post; none needs a live network call or a real API key, matching
this project's existing convention for core.semantic_judge's tests.
"""
from unittest.mock import patch

import pytest
import requests

from core.llm_providers import (
    AnthropicCompatibleProvider,
    LLMProviderError,
    LLMResponse,
    OllamaProvider,
    OpenAICompatibleProvider,
    get_provider,
)


class MockResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self.json_data = json_data
        self.status_code = status_code
        self.text = text or str(json_data)

    def json(self):
        return self.json_data


MESSAGES = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

def test_ollama_success():
    provider = OllamaProvider(base_url="http://x/api/chat", default_model="llama3.2")
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"message": {"content": "hi there"}})) as mock_post:
        result = provider.complete(MESSAGES)

    assert isinstance(result, LLMResponse)
    assert result.content == "hi there"
    assert result.model == "llama3.2"
    assert result.provider == "ollama"
    assert result.usage is None
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"]["model"] == "llama3.2"


def test_ollama_explicit_model_overrides_default():
    provider = OllamaProvider(base_url="http://x", default_model="llama3.2")
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"message": {"content": "hi"}})) as mock_post:
        result = provider.complete(MESSAGES, model="mistral")
    assert result.model == "mistral"
    assert mock_post.call_args.kwargs["json"]["model"] == "mistral"


def test_ollama_non_200_raises():
    provider = OllamaProvider(base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({}, status_code=500, text="boom")):
        with pytest.raises(LLMProviderError):
            provider.complete(MESSAGES)


def test_ollama_malformed_response_raises():
    provider = OllamaProvider(base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", return_value=MockResponse({"unexpected": "shape"})):
        with pytest.raises(LLMProviderError):
            provider.complete(MESSAGES)


def test_ollama_network_error_raises_llm_provider_error_not_requests_exception():
    """Callers should only ever need to catch one exception type."""
    provider = OllamaProvider(base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(LLMProviderError):
            provider.complete(MESSAGES)


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider
# ---------------------------------------------------------------------------

def test_openai_compatible_success():
    provider = OpenAICompatibleProvider(api_key="sk-test", base_url="http://x/v1", default_model="gpt-4o-mini")
    mock_body = {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "hi there"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)) as mock_post:
        result = provider.complete(MESSAGES)

    assert result.content == "hi there"
    assert result.provider == "openai_compatible"
    assert result.usage == {"prompt_tokens": 5, "completion_tokens": 3}
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert mock_post.call_args.args[0] == "http://x/v1/chat/completions"


def test_openai_compatible_missing_api_key_raises_without_network_call():
    provider = OpenAICompatibleProvider(api_key="", base_url="http://x/v1", default_model="gpt-4o-mini")
    with patch("core.llm_providers.requests.post") as mock_post:
        with pytest.raises(LLMProviderError, match="API key"):
            provider.complete(MESSAGES)
        mock_post.assert_not_called()


def test_openai_compatible_missing_model_raises_without_network_call():
    provider = OpenAICompatibleProvider(api_key="sk-test", base_url="http://x/v1", default_model="")
    with patch("core.llm_providers.requests.post") as mock_post:
        with pytest.raises(LLMProviderError, match="model"):
            provider.complete(MESSAGES)
        mock_post.assert_not_called()


def test_openai_compatible_non_200_raises():
    provider = OpenAICompatibleProvider(api_key="sk-test", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", return_value=MockResponse({}, status_code=401, text="unauthorized")):
        with pytest.raises(LLMProviderError):
            provider.complete(MESSAGES)


def test_openai_compatible_malformed_response_raises():
    provider = OpenAICompatibleProvider(api_key="sk-test", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", return_value=MockResponse({"choices": []})):
        with pytest.raises(LLMProviderError):
            provider.complete(MESSAGES)


def test_openai_compatible_base_url_trailing_slash_normalised():
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://x/v1/", default_model="m")
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"choices": [{"message": {"content": "hi"}}]})) as mock_post:
        provider.complete(MESSAGES)
    assert mock_post.call_args.args[0] == "http://x/v1/chat/completions"


# ---------------------------------------------------------------------------
# AnthropicCompatibleProvider
# ---------------------------------------------------------------------------

def test_anthropic_compatible_success():
    provider = AnthropicCompatibleProvider(api_key="ant-test", base_url="http://x/v1", default_model="claude")
    mock_body = {
        "model": "claude",
        "content": [{"type": "text", "text": "hi there"}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)) as mock_post:
        result = provider.complete(MESSAGES)

    assert result.content == "hi there"
    assert result.provider == "anthropic_compatible"
    assert result.usage == {"input_tokens": 5, "output_tokens": 3}
    assert mock_post.call_args.kwargs["headers"]["x-api-key"] == "ant-test"
    assert "anthropic-version" in mock_post.call_args.kwargs["headers"]


def test_anthropic_compatible_extracts_system_message_to_top_level_field():
    """The one real wire-format difference this abstraction has to paper
    over: Anthropic takes `system` as a top-level field, not a
    role=system message in the list."""
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"content": [{"text": "hi"}]})) as mock_post:
        provider.complete(messages)

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["system"] == "You are a helpful assistant."
    assert sent_payload["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_compatible_no_system_message_omits_the_field():
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"content": [{"text": "hi"}]})) as mock_post:
        provider.complete(MESSAGES)
    assert "system" not in mock_post.call_args.kwargs["json"]


def test_anthropic_compatible_missing_api_key_raises_without_network_call():
    provider = AnthropicCompatibleProvider(api_key="", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post") as mock_post:
        with pytest.raises(LLMProviderError, match="API key"):
            provider.complete(MESSAGES)
        mock_post.assert_not_called()


def test_anthropic_compatible_non_200_raises():
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", return_value=MockResponse({}, status_code=529, text="overloaded")):
        with pytest.raises(LLMProviderError):
            provider.complete(MESSAGES)


def test_anthropic_compatible_malformed_response_raises():
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", return_value=MockResponse({"content": []})):
        with pytest.raises(LLMProviderError):
            provider.complete(MESSAGES)


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------

def test_get_provider_returns_correct_instances():
    assert isinstance(get_provider("ollama"), OllamaProvider)
    assert isinstance(get_provider("openai_compatible"), OpenAICompatibleProvider)
    assert isinstance(get_provider("anthropic_compatible"), AnthropicCompatibleProvider)


def test_get_provider_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        get_provider("nonexistent_provider")


# ---------------------------------------------------------------------------
# Normalisation: all three providers produce the same response shape
# ---------------------------------------------------------------------------

def test_all_three_providers_produce_the_same_response_shape():
    ollama = OllamaProvider(base_url="http://x", default_model="m")
    openai_p = OpenAICompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    anthropic_p = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")

    with patch("core.llm_providers.requests.post", return_value=MockResponse({"message": {"content": "c"}})):
        r1 = ollama.complete(MESSAGES)
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"choices": [{"message": {"content": "c"}}]})):
        r2 = openai_p.complete(MESSAGES)
    with patch("core.llm_providers.requests.post", return_value=MockResponse({"content": [{"text": "c"}]})):
        r3 = anthropic_p.complete(MESSAGES)

    for r in (r1, r2, r3):
        assert isinstance(r, LLMResponse)
        assert r.content == "c"
        assert isinstance(r.provider, str)
