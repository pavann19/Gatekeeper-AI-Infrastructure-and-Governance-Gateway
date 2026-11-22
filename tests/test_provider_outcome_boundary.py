"""Uncertain provider execution must not trigger a second completion.

Local synthetic responses and a bounded worker fixture; no external provider.
"""
import threading
from unittest.mock import Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from api.main import app
from core.llm_providers import (AnthropicCompatibleProvider, LLMProviderError,
                                LLMResponse, OllamaProvider, OpenAICompatibleProvider)

pytestmark = pytest.mark.usefixtures("provisioned_tenants")


def provider_for(name):
    if name == "ollama":
        return OllamaProvider(base_url="http://synthetic.invalid", default_model="mock")
    cls = OpenAICompatibleProvider if name == "openai" else AnthropicCompatibleProvider
    return cls(api_key="synthetic-only", base_url="http://synthetic.invalid", default_model="mock")


@pytest.mark.parametrize("provider_name", ["ollama", "openai", "anthropic"])
@pytest.mark.parametrize("failure", ["timeout", "connection", "server_error", "malformed_json",
                                     "null_body", "null_content", "missing_content"])
def test_uncertain_adapter_failure_is_distinguished_from_safe_preflight_failure(provider_name, failure):
    response = Mock(status_code=500 if failure == "server_error" else 200)
    body = {"message": {"content": None}, "choices": [{"message": {"content": None}}], "content": [{"text": None}]}
    response.json.return_value = None if failure == "null_body" else {} if failure == "missing_content" else body
    if failure == "malformed_json":
        response.json.side_effect = ValueError("synthetic malformed json")
    error = requests.ReadTimeout("synthetic timeout") if failure == "timeout" else (
        requests.ConnectionError("synthetic disconnect") if failure == "connection" else None)
    with patch("core.llm_providers.requests.post", return_value=response, side_effect=error):
        with pytest.raises(LLMProviderError) as raised:
            provider_for(provider_name).complete([{"role": "user", "content": "synthetic"}])
    assert getattr(raised.value, "outcome_unknown", False)


@pytest.mark.parametrize("failure", ["transport_timeout", "worker_timeout"])
def test_gateway_never_dispatches_fallback_after_uncertain_execution(monkeypatch, failure):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr("api.main.settings.GATEWAY_TOKEN_QUOTA_ENABLED", False)
    monkeypatch.setattr("api.main.settings.GATEWAY_FALLBACK_PROVIDERS", "openai_compatible")
    monkeypatch.setattr("api.main.settings.GATEWAY_TIMEOUT_SECONDS", 0.1 if failure == "worker_timeout" else 10)
    started, release = threading.Event(), threading.Event()

    def slow_completion(*args):
        started.set()
        assert release.wait(timeout=5), "Synthetic worker was not released"
        return LLMResponse(content="synthetic", model="mock", provider="ollama")

    with patch("api.main.assess_risk", return_value=("LOW", {"source": "mock"})), \
         patch("api.main._redact_or_fail", return_value=("synthetic", {})), \
         patch("core.output_guardrails.assess_output", return_value=("ALLOW", {"clean_response": "synthetic"})), \
         patch("core.llm_providers.OpenAICompatibleProvider.complete", return_value=LLMResponse(
             content="synthetic", model="mock", provider="openai_compatible")) as fallback:
        try:
            if failure == "transport_timeout":
                with patch("core.llm_providers.requests.post", side_effect=requests.ReadTimeout("synthetic")):
                    response = TestClient(app).post("/api/v1/gateway/chat", json={"prompt": "synthetic"})
            else:
                with patch("core.llm_providers.OllamaProvider.complete", side_effect=slow_completion):
                    response = TestClient(app).post("/api/v1/gateway/chat", json={"prompt": "synthetic"})
                assert started.is_set()
            fallback.assert_not_called()
            assert response.status_code == 503
            assert "unknown" in response.json()["detail"].lower()
            assert "Retry-After" not in response.headers
        finally:
            release.set()
