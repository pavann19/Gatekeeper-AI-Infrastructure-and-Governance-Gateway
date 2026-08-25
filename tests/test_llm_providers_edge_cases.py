"""
Additional edge-case tests for core.llm_providers, written to complement
(not duplicate) tests/test_llm_providers.py. Focus areas: timeout-specific
error handling, usage/token-count shapes as actually consumed by
core.token_quota.extract_total_tokens, list_provider_names() stability,
request payload/header shape details, and get_provider() instance
independence. Every test mocks requests.post; no live network call.
"""
from unittest.mock import patch

import pytest
import requests

from core.llm_providers import (
    AnthropicCompatibleProvider,
    LLMProviderError,
    OllamaProvider,
    OpenAICompatibleProvider,
    _PROVIDER_CLASSES,
    get_provider,
    list_provider_names,
)
from core.token_quota import extract_total_tokens


class MockResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self.json_data = json_data
        self.status_code = status_code
        self.text = text or str(json_data)

    def json(self):
        return self.json_data


MESSAGES = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# list_provider_names() / get_provider() registry stability
# ---------------------------------------------------------------------------

def test_list_provider_names_matches_registered_classes_exactly():
    assert list_provider_names() == sorted(_PROVIDER_CLASSES.keys())


def test_list_provider_names_is_the_expected_stable_set():
    assert list_provider_names() == ["anthropic_compatible", "ollama", "openai_compatible"]


def test_list_provider_names_returns_a_list_of_str():
    names = list_provider_names()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


def test_get_provider_unknown_name_error_message_lists_available_providers():
    with pytest.raises(KeyError) as exc_info:
        get_provider("bogus")
    # KeyError str() wraps the message in quotes; check the substance.
    message = str(exc_info.value)
    assert "bogus" in message
    for name in list_provider_names():
        assert name in message


def test_get_provider_returns_a_fresh_instance_each_call():
    p1 = get_provider("ollama")
    p2 = get_provider("ollama")
    assert p1 is not p2
    assert isinstance(p1, OllamaProvider) and isinstance(p2, OllamaProvider)


# ---------------------------------------------------------------------------
# Timeout-specific error handling (distinct from generic ConnectionError)
# ---------------------------------------------------------------------------

def test_ollama_timeout_raises_llm_provider_error():
    provider = OllamaProvider(base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", side_effect=requests.Timeout("timed out")):
        with pytest.raises(LLMProviderError, match="Timeout"):
            provider.complete(MESSAGES)


def test_openai_compatible_timeout_raises_llm_provider_error():
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", side_effect=requests.Timeout("timed out")):
        with pytest.raises(LLMProviderError, match="Timeout"):
            provider.complete(MESSAGES)


def test_anthropic_compatible_timeout_raises_llm_provider_error():
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", side_effect=requests.Timeout("timed out")):
        with pytest.raises(LLMProviderError, match="Timeout"):
            provider.complete(MESSAGES)


def test_timeout_argument_is_forwarded_to_requests_post():
    """The caller-supplied timeout must actually reach requests.post, not
    just the module-level default — otherwise a caller's shorter deadline
    is silently ignored."""
    provider = OllamaProvider(base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"message": {"content": "hi"}})) as mock_post:
        provider.complete(MESSAGES, timeout=5)
    assert mock_post.call_args.kwargs["timeout"] == 5


def test_openai_compatible_connection_error_raises_llm_provider_error_with_type_name():
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(LLMProviderError, match="ConnectionError"):
            provider.complete(MESSAGES)


def test_anthropic_compatible_connection_error_raises_llm_provider_error_with_type_name():
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(LLMProviderError, match="ConnectionError"):
            provider.complete(MESSAGES)


# ---------------------------------------------------------------------------
# Usage/token-count extraction correctness, feeding core.token_quota
# ---------------------------------------------------------------------------

def test_ollama_usage_is_none_and_extracts_to_zero_tokens():
    """Ollama's /api/chat reports no usage; extract_total_tokens must
    degrade to 0, not raise, so quota accounting doesn't blow up."""
    provider = OllamaProvider(base_url="http://x", default_model="m")
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"message": {"content": "hi"}})):
        result = provider.complete(MESSAGES)
    assert result.usage is None
    assert extract_total_tokens(result.usage) == 0


def test_openai_compatible_usage_extracts_total_tokens_field():
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    mock_body = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)):
        result = provider.complete(MESSAGES)
    assert result.usage == {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
    assert extract_total_tokens(result.usage) == 20


def test_openai_compatible_usage_missing_entirely_extracts_to_zero():
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    mock_body = {"choices": [{"message": {"content": "hi"}}]}
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)):
        result = provider.complete(MESSAGES)
    assert result.usage is None
    assert extract_total_tokens(result.usage) == 0


def test_anthropic_compatible_usage_sums_input_and_output_tokens():
    """Anthropic has no combined field -- extract_total_tokens must sum
    input_tokens + output_tokens itself."""
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    mock_body = {"content": [{"text": "hi"}], "usage": {"input_tokens": 15, "output_tokens": 25}}
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)):
        result = provider.complete(MESSAGES)
    assert result.usage == {"input_tokens": 15, "output_tokens": 25}
    assert extract_total_tokens(result.usage) == 40


def test_anthropic_compatible_usage_with_only_output_tokens_present():
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    mock_body = {"content": [{"text": "hi"}], "usage": {"output_tokens": 7}}
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)):
        result = provider.complete(MESSAGES)
    assert extract_total_tokens(result.usage) == 7


# ---------------------------------------------------------------------------
# Response content extraction from real-shaped multi-element bodies
# ---------------------------------------------------------------------------

def test_anthropic_compatible_uses_first_content_block_when_multiple_present():
    """Real Anthropic responses can carry multiple content blocks (e.g. a
    tool_use block after text); this module only ever reads block [0]."""
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    mock_body = {
        "content": [
            {"type": "text", "text": "first block text"},
            {"type": "text", "text": "second block text"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)):
        result = provider.complete(MESSAGES)
    assert result.content == "first block text"


def test_openai_compatible_uses_first_choice_when_multiple_present():
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    mock_body = {
        "choices": [
            {"message": {"content": "choice zero"}},
            {"message": {"content": "choice one"}},
        ],
    }
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)):
        result = provider.complete(MESSAGES)
    assert result.content == "choice one" or result.content == "choice zero"
    # This module reads index [0] specifically -- pin that down exactly.
    assert result.content == "choice zero"


def test_openai_compatible_model_falls_back_to_requested_when_absent_from_response():
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", default_model="requested-model")
    mock_body = {"choices": [{"message": {"content": "hi"}}]}  # no "model" key
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)):
        result = provider.complete(MESSAGES)
    assert result.model == "requested-model"


def test_anthropic_compatible_model_falls_back_to_requested_when_absent_from_response():
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="requested-model")
    mock_body = {"content": [{"text": "hi"}]}  # no "model" key
    with patch("core.llm_providers.requests.post", return_value=MockResponse(mock_body)):
        result = provider.complete(MESSAGES)
    assert result.model == "requested-model"


# ---------------------------------------------------------------------------
# Request payload/header shape correctness
# ---------------------------------------------------------------------------

def test_ollama_payload_sets_stream_false_and_full_message_list():
    provider = OllamaProvider(base_url="http://x", default_model="m")
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"message": {"content": "hi"}})) as mock_post:
        provider.complete(messages)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream"] is False
    assert payload["messages"] == messages  # unlike Anthropic, no filtering


def test_openai_compatible_content_type_header_and_payload_shape():
    provider = OpenAICompatibleProvider(api_key="sk-abc", base_url="http://x", default_model="gpt-4o-mini")
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"choices": [{"message": {"content": "hi"}}]})) as mock_post:
        provider.complete(MESSAGES)
    headers = mock_post.call_args.kwargs["headers"]
    payload = mock_post.call_args.kwargs["json"]
    assert headers["Content-Type"] == "application/json"
    assert payload == {"model": "gpt-4o-mini", "messages": MESSAGES}


def test_anthropic_compatible_payload_includes_max_tokens_and_version_header():
    provider = AnthropicCompatibleProvider(
        api_key="ant-abc", base_url="http://x", default_model="claude-x",
        anthropic_version="2023-06-01",
    )
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"content": [{"text": "hi"}]})) as mock_post:
        provider.complete(MESSAGES)
    payload = mock_post.call_args.kwargs["json"]
    headers = mock_post.call_args.kwargs["headers"]
    assert payload["max_tokens"] == 4096
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["Content-Type"] == "application/json"


def test_anthropic_compatible_multiple_system_messages_are_joined():
    provider = AnthropicCompatibleProvider(api_key="k", base_url="http://x", default_model="m")
    messages = [
        {"role": "system", "content": "first instruction"},
        {"role": "system", "content": "second instruction"},
        {"role": "user", "content": "hi"},
    ]
    with patch("core.llm_providers.requests.post",
              return_value=MockResponse({"content": [{"text": "hi"}]})) as mock_post:
        provider.complete(messages)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["system"] == "first instruction\n\nsecond instruction"
