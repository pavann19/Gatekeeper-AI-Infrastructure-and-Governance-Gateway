"""
Unit tests for scripts/exercise_live_providers.py.
"""
from unittest import mock
from core.llm_providers import LLMProviderError, LLMResponse
from scripts.exercise_live_providers import exercise_provider


def test_exercise_provider_success():
    fake_resp = LLMResponse(
        content="Hello from test model!",
        model="gpt-4o-mini",
        provider="openai_compatible",
        usage={"total_tokens": 42},
    )

    with mock.patch("scripts.exercise_live_providers.get_provider") as mock_get:
        mock_instance = mock.MagicMock()
        mock_instance.complete.return_value = fake_resp
        mock_instance.base_url = "https://api.openai.com/v1"
        mock_instance.api_key = "sk-test"
        mock_get.return_value = mock_instance

        res = exercise_provider("openai_compatible", prompt="test prompt")

    assert res["status"] == "success"
    assert res["model"] == "gpt-4o-mini"
    assert res["tokens_used"] == 42
    assert "Hello from test" in res["content_preview"]


def test_exercise_provider_failure_handled():
    with mock.patch("scripts.exercise_live_providers.get_provider") as mock_get:
        mock_instance = mock.MagicMock()
        mock_instance.complete.side_effect = LLMProviderError("Connection refused")
        mock_get.return_value = mock_instance

        res = exercise_provider("ollama")

    assert res["status"] == "failed"
    assert "Connection refused" in res["error"]


def test_exercise_provider_unknown_provider():
    res = exercise_provider("unknown_vendor")
    assert res["status"] == "error"
    assert "Failed to instantiate provider" in res["error"]
