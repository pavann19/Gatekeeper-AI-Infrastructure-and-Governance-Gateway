"""Regression contracts for budget configuration and completed-call accounting.

All provider/output failures are synthetic; tests use provisioned temporary
tenants and an injected in-memory tracker, never a live account or model.
"""
import asyncio
import json
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.main import app
from core.admission import require_admission
from core.auth import Principal
from core.config import Settings
from core.llm_providers import LLMResponse, OllamaProvider
from core.rate_limit import LocalRateLimiter, RedisRateLimiter
from core.tenancy import TenantStore
from core.token_quota import LocalTokenQuotaTracker, extract_total_tokens


@pytest.mark.parametrize("field,value", [
    ("token_quota_daily", value) for value in
    (0.5, -0.5, 1.9, True, False, "0.5", "unlimited", -1, float("inf"), float("nan"))
] + [
    ("rate_limit_rpm", value) for value in
    (float("nan"), float("inf"), -float("inf"), "nan", "inf", True, False, 0, -1, "bad")
])
def test_invalid_tenant_budget_denies_only_affected_tenant(tmp_path, field, value):
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({"bad": {field: value}, "good": {field: 100}}), encoding="utf-8")
    store = TenantStore(str(path)).load()
    bad = store.get("bad")
    with pytest.raises(HTTPException) as denied:
        require_admission(Principal(tenant="bad"), bad)
    assert denied.value.status_code == 403
    assert not store.get("good").suspended


@pytest.mark.parametrize("quota", [0, 1, 100, "0", "100"])
def test_explicit_valid_quota_remains_supported(tmp_path, quota):
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({"tenant": {"token_quota_daily": quota}}), encoding="utf-8")
    tenant = TenantStore(str(path)).load().get("tenant")
    assert not tenant.suspended
    assert tenant.token_quota_daily == int(quota)


def test_local_rate_limit_reduction_clamps_existing_burst_without_clock_advance():
    limiter = LocalRateLimiter("downgrade")
    assert limiter.check("tenant", 100, 1, now=100)[0]
    results = [limiter.check("tenant", 5, 1, now=100)[0] for _ in range(100)]
    assert sum(results) == 5


@pytest.mark.parametrize("field", ["RATE_LIMIT_AUTHENTICATED_RPM", "RATE_LIMIT_ANONYMOUS_RPM",
                                  "RATE_LIMIT_BURST_SECONDS", "ASSESS_TIMEOUT_SECONDS",
                                  "GATEWAY_TIMEOUT_SECONDS", "STARTUP_WARMUP_TIMEOUT_SECONDS"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_global_budgets_are_rejected(field, value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("field", ["capacity", "refill_per_second", "now"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_backend_budgets_deny_without_redis_call(field, value):
    client = Mock()
    for limiter in (LocalRateLimiter("invalid"), RedisRateLimiter(client, "invalid")):
        args = {"capacity": 5, "refill_per_second": 1, "now": 100, field: value}
        assert limiter.check("tenant", **args)[0] is False
    client.register_script.return_value.assert_not_called()


@pytest.mark.parametrize("usage,expected", [
    ({"input_tokens": 10, "output_tokens": "bad"}, 10),
    ({"input_tokens": "bad", "output_tokens": 7}, 7),
    ({"input_tokens": -100, "output_tokens": 7}, 7),
    ({"total_tokens": -1, "input_tokens": 10, "output_tokens": 7}, 17),
    ({"total_tokens": True, "input_tokens": 10, "output_tokens": 7}, 17),
    ({"total_tokens": 1, "input_tokens": 10, "output_tokens": 7}, 17),
    ({"input_tokens": True, "output_tokens": False}, 0),
])
def test_malformed_usage_cannot_crash_or_reduce_known_usage(usage, expected):
    assert extract_total_tokens(usage) == expected


def test_ollama_reported_usage_is_not_discarded():
    response = Mock(status_code=200)
    response.json.return_value = {"message": {"content": "synthetic response"},
                                  "prompt_eval_count": 11, "eval_count": 18}
    with patch("core.llm_providers.requests.post", return_value=response):
        result = OllamaProvider().complete([{"role": "user", "content": "synthetic"}])
    assert result.usage == {"input_tokens": 11, "output_tokens": 18, "total_tokens": 29}


@pytest.mark.parametrize("failure", ["output_timeout", "output_exception", "output_audit_failure"])
def test_completed_provider_usage_survives_downstream_failure(monkeypatch, provisioned_tenants, failure):
    tracker = LocalTokenQuotaTracker()
    monkeypatch.setattr("api.main.gateway_token_quota", tracker)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr("api.main.settings.GATEWAY_TOKEN_QUOTA_ENABLED", True)
    monkeypatch.setattr("api.main.settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT", 77)
    output = Mock(return_value=("ALLOW", {"clean_response": "synthetic"}))
    if failure == "output_timeout":
        output.side_effect = asyncio.TimeoutError()
    elif failure == "output_exception":
        output.side_effect = RuntimeError("synthetic output failure")
    with patch("api.main.assess_risk", return_value=("LOW", {"source": "mock"})), \
         patch("api.main._redact_or_fail", return_value=("synthetic", {})), \
         patch("core.llm_providers.OllamaProvider.complete", return_value=LLMResponse(
             content="synthetic", model="mock", provider="ollama", usage={"total_tokens": 77})) as complete, \
         patch("core.output_guardrails.assess_output", output), \
         patch("api.main.log_output_event", side_effect=OSError("synthetic audit failure")
               if failure == "output_audit_failure" else None):
        client = TestClient(app, raise_server_exceptions=False)
        first = client.post("/api/v1/gateway/chat", json={"prompt": "synthetic"})
        assert first.status_code >= 500
        assert tracker.usage_today("default") == 77
        second = client.post("/api/v1/gateway/chat", json={"prompt": "synthetic"})
        assert second.status_code == 429
        assert complete.call_count == 1
