"""
Tests for POST /api/v1/gateway/chat (Phase 5: Real LLM Gateway — request
forwarding + response interception). Every provider call is mocked; no
live network or API key needed.
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import auth as auth_mod
from core import policy as policy_mod
from core.auth import KeyStore, generate_key, hash_key
from core.llm_providers import LLMProviderError, LLMResponse
from core.policy import PolicyStore

client = TestClient(app)

LOW_RISK = ("LOW", {"semantic_score": 0.1, "source": "mock"})
HIGH_RISK = ("HIGH", {"semantic_score": 0.9, "source": "mock"})
MEDIUM_RISK = ("MEDIUM", {"semantic_score": 0.5, "source": "mock"})


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


@pytest.fixture
def key_store(tmp_path, monkeypatch):
    path = tmp_path / "api_keys.json"
    store = {}

    def issue(capability="GENERAL", tenant="default", key_id="test-key"):
        plaintext = generate_key()
        store[hash_key(plaintext)] = {"capability": capability, "tenant": tenant, "key_id": key_id}
        path.write_text(json.dumps(store), encoding="utf-8")
        monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
        return plaintext

    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
    return issue


@pytest.fixture
def review_policy(tmp_path, monkeypatch):
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {
            "GENERAL": {"HIGH": "BLOCK", "MEDIUM": "REVIEW", "LOW": "ALLOW"},
        }}},
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))


CLEAN_OUTPUT = ("ALLOW", {"source": "clean_pass", "clean_response": "Here is your answer."})
DIRTY_OUTPUT = ("BLOCK", {"source": "secret_leakage", "secrets_detected": True})


# ---------------------------------------------------------------------------
# Happy path: input allowed -> provider called -> output allowed
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_happy_path_forwards_to_provider_and_returns_content(mock_assess):
    mock_llm_response = LLMResponse(content="Here is your answer.", model="llama3.2", provider="ollama")
    with patch("core.llm_providers.OllamaProvider.complete", return_value=mock_llm_response), \
         patch("core.output_guardrails.assess_output", return_value=CLEAN_OUTPUT):
        response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "ALLOW"
    assert body["content"] == "Here is your answer."
    assert body["provider"] == "ollama"
    assert body["model"] == "llama3.2"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_explicit_provider_and_model_are_honoured(mock_assess):
    mock_llm_response = LLMResponse(content="hi", model="gpt-4o-mini", provider="openai_compatible")
    with patch("core.llm_providers.OpenAICompatibleProvider.complete", return_value=mock_llm_response) as mock_complete, \
         patch("core.output_guardrails.assess_output", return_value=("ALLOW", {"clean_response": "hi"})):
        response = client.post("/api/v1/gateway/chat", json={
            "prompt": "hello", "provider": "openai_compatible", "model": "gpt-4o-mini",
        })

    assert response.json()["provider"] == "openai_compatible"
    mock_complete.assert_called_once()
    assert mock_complete.call_args.args[1] == "gpt-4o-mini"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_unknown_provider_is_422(mock_assess):
    response = client.post("/api/v1/gateway/chat", json={"prompt": "hello", "provider": "nonexistent"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Input guardrails stop the provider from ever being called
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=HIGH_RISK)
def test_high_risk_input_never_reaches_the_provider(mock_assess):
    with patch("core.llm_providers.OllamaProvider.complete") as mock_complete:
        response = client.post("/api/v1/gateway/chat", json={"prompt": "dangerous"})
        mock_complete.assert_not_called()

    assert response.status_code == 200
    assert response.json()["decision"] == "BLOCK"
    assert response.json()["content"] is None


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_review_decision_never_reaches_the_provider(mock_assess, review_policy):
    with patch("core.llm_providers.OllamaProvider.complete") as mock_complete:
        response = client.post("/api/v1/gateway/chat", json={"prompt": "ambiguous"})
        mock_complete.assert_not_called()

    body = response.json()
    assert body["decision"] == "REVIEW"
    assert body["review_id"] is not None
    assert body["content"] is None


# ---------------------------------------------------------------------------
# Output guardrails on what the provider actually returned
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_output_guardrails_block_a_leaked_secret_from_the_provider(mock_assess):
    mock_llm_response = LLMResponse(content="key: AKIAABCDEFGHIJKLMNOP", model="m", provider="ollama")
    with patch("core.llm_providers.OllamaProvider.complete", return_value=mock_llm_response), \
         patch("core.output_guardrails.assess_output", return_value=DIRTY_OUTPUT):
        response = client.post("/api/v1/gateway/chat", json={"prompt": "give me a key"})

    assert response.json()["decision"] == "BLOCK"
    assert response.json()["content"] is None


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_clean_prompt_with_dirty_response_still_blocks_the_final_decision(mock_assess):
    """The whole point of output interception: a clean input doesn't
    guarantee a safe response, and the FINAL decision must reflect that."""
    mock_llm_response = LLMResponse(content="toxic content", model="m", provider="ollama")
    with patch("core.llm_providers.OllamaProvider.complete", return_value=mock_llm_response), \
         patch("core.output_guardrails.assess_output",
              return_value=("BLOCK", {"toxicity_detected": True, "source": "semantic_judge_output"})):
        response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
    assert response.json()["decision"] == "BLOCK"


# ---------------------------------------------------------------------------
# Provider failure and timeout handling
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_provider_error_returns_502_not_a_fabricated_decision(mock_assess):
    with patch("core.llm_providers.OllamaProvider.complete", side_effect=LLMProviderError("boom")):
        response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
    assert response.status_code == 502


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_provider_timeout_returns_503(mock_assess, monkeypatch):
    monkeypatch.setattr("api.main.settings.GATEWAY_TIMEOUT_SECONDS", 0.05)
    import time as _time
    with patch("core.llm_providers.OllamaProvider.complete", side_effect=lambda m, mo=None: _time.sleep(1.0)):
        response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Audit trail: a distinct gateway_call event, separate from input/output
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_successful_call_logs_a_distinct_gateway_event(mock_assess):
    mock_llm_response = LLMResponse(content="hi", model="m", provider="ollama", usage={"total_tokens": 10})
    with patch("core.llm_providers.OllamaProvider.complete", return_value=mock_llm_response), \
         patch("core.output_guardrails.assess_output", return_value=("ALLOW", {"clean_response": "hi"})), \
         patch("api.main.log_gateway_event") as mock_log_gateway, \
         patch("api.main.log_event") as mock_log_input, \
         patch("api.main.log_output_event") as mock_log_output:
        client.post("/api/v1/gateway/chat", json={"prompt": "hello"})

    mock_log_input.assert_called_once()
    mock_log_output.assert_called_once()
    mock_log_gateway.assert_called_once()
    call_kwargs = mock_log_gateway.call_args
    assert call_kwargs.kwargs["success"] is True
    assert call_kwargs.kwargs["usage"] == {"total_tokens": 10}


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_provider_failure_still_logs_a_gateway_event(mock_assess):
    with patch("core.llm_providers.OllamaProvider.complete", side_effect=LLMProviderError("boom")), \
         patch("api.main.log_gateway_event") as mock_log_gateway:
        client.post("/api/v1/gateway/chat", json={"prompt": "hello"})

    mock_log_gateway.assert_called_once()
    assert mock_log_gateway.call_args.kwargs["success"] is False


@patch("api.main.assess_risk", return_value=HIGH_RISK)
def test_blocked_input_does_not_log_a_gateway_event(mock_assess):
    """No provider call happened -- there is nothing for a gateway_call
    event to describe. The input event still fires (log_event)."""
    with patch("api.main.log_gateway_event") as mock_log_gateway:
        client.post("/api/v1/gateway/chat", json={"prompt": "dangerous"})
    mock_log_gateway.assert_not_called()


# ---------------------------------------------------------------------------
# Auth boundary, identical to /api/v1/assess
# ---------------------------------------------------------------------------

def test_requires_auth_when_auth_mode_required(monkeypatch):
    monkeypatch.setattr("api.main.settings.AUTH_MODE", "required")
    response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
    assert response.status_code == 401


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_valid_key_authenticates_the_call(mock_assess, key_store):
    key = key_store(capability="GENERAL")
    mock_llm_response = LLMResponse(content="hi", model="m", provider="ollama")
    with patch("core.llm_providers.OllamaProvider.complete", return_value=mock_llm_response), \
         patch("core.output_guardrails.assess_output", return_value=("ALLOW", {"clean_response": "hi"})):
        response = client.post(
            "/api/v1/gateway/chat", json={"prompt": "hello"},
            headers={"Authorization": f"Bearer {key}"},
        )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Token accounting (Phase 5): enforced against usage already recorded from
# past calls, never a pre-flight estimate.
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_quota_exceeded_returns_429_and_never_calls_the_provider(mock_assess, monkeypatch):
    from core.token_quota import gateway_token_quota
    monkeypatch.setattr("api.main.settings.GATEWAY_TOKEN_QUOTA_ENABLED", True)
    monkeypatch.setattr("api.main.settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT", 100)
    gateway_token_quota.reset()
    gateway_token_quota.record("default", 150)
    try:
        with patch("core.llm_providers.OllamaProvider.complete") as mock_complete:
            response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
            mock_complete.assert_not_called()
        assert response.status_code == 429
    finally:
        gateway_token_quota.reset()


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_successful_call_records_usage_against_the_tenant_quota(mock_assess, monkeypatch):
    from core.token_quota import gateway_token_quota
    monkeypatch.setattr("api.main.settings.GATEWAY_TOKEN_QUOTA_ENABLED", True)
    monkeypatch.setattr("api.main.settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT", 1000)
    gateway_token_quota.reset()
    mock_llm_response = LLMResponse(content="hi", model="m", provider="ollama",
                                    usage={"total_tokens": 77})
    try:
        with patch("core.llm_providers.OllamaProvider.complete", return_value=mock_llm_response), \
             patch("core.output_guardrails.assess_output", return_value=("ALLOW", {"clean_response": "hi"})):
            response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
        assert response.status_code == 200
        assert gateway_token_quota.usage_today("default") == 77
    finally:
        gateway_token_quota.reset()


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_quota_disabled_by_default_even_with_prior_usage(mock_assess, monkeypatch):
    from core.token_quota import gateway_token_quota
    monkeypatch.setattr("api.main.settings.GATEWAY_TOKEN_QUOTA_ENABLED", False)
    gateway_token_quota.reset()
    gateway_token_quota.record("default", 10_000_000)
    mock_llm_response = LLMResponse(content="hi", model="m", provider="ollama")
    try:
        with patch("core.llm_providers.OllamaProvider.complete", return_value=mock_llm_response), \
             patch("core.output_guardrails.assess_output", return_value=("ALLOW", {"clean_response": "hi"})):
            response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
        assert response.status_code == 200
    finally:
        gateway_token_quota.reset()


# ---------------------------------------------------------------------------
# Cross-provider fallback (Phase 5): only when the caller left `provider`
# unset -- an explicit choice is never second-guessed.
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_fallback_provider_is_used_when_primary_fails(mock_assess, monkeypatch):
    monkeypatch.setattr("api.main.settings.GATEWAY_FALLBACK_PROVIDERS", "openai_compatible")
    fallback_response = LLMResponse(content="from fallback", model="gpt-4o-mini", provider="openai_compatible")
    with patch("core.llm_providers.OllamaProvider.complete", side_effect=LLMProviderError("primary down")), \
         patch("core.llm_providers.OpenAICompatibleProvider.complete", return_value=fallback_response), \
         patch("core.output_guardrails.assess_output", return_value=("ALLOW", {"clean_response": "from fallback"})):
        response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai_compatible"
    assert body["details"]["gateway_fallback_used"] is True
    assert body["details"]["gateway_fallback_from"] == "ollama"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_fallback_not_used_when_caller_explicitly_named_a_provider(mock_assess, monkeypatch):
    monkeypatch.setattr("api.main.settings.GATEWAY_FALLBACK_PROVIDERS", "openai_compatible")
    with patch("core.llm_providers.OllamaProvider.complete", side_effect=LLMProviderError("primary down")) as mock_ollama, \
         patch("core.llm_providers.OpenAICompatibleProvider.complete") as mock_openai:
        response = client.post("/api/v1/gateway/chat", json={"prompt": "hello", "provider": "ollama"})

    mock_ollama.assert_called_once()
    mock_openai.assert_not_called()
    assert response.status_code == 502


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_all_providers_in_chain_failing_returns_the_last_error(mock_assess, monkeypatch):
    monkeypatch.setattr("api.main.settings.GATEWAY_FALLBACK_PROVIDERS", "openai_compatible")
    with patch("core.llm_providers.OllamaProvider.complete", side_effect=LLMProviderError("primary down")), \
         patch("core.llm_providers.OpenAICompatibleProvider.complete", side_effect=LLMProviderError("fallback down too")):
        response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})

    assert response.status_code == 502
    assert "fallback down too" in response.json()["detail"]
