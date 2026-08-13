"""
Endpoint-level enforcement of tenant resolution: suspension -> 403, SLA ->
rate limit override, and the audit/metrics fields it's supposed to populate.

API-level rather than unit tests, same reasoning as test_request_limits.py:
resolve_tenant() working correctly in isolation (tests/test_tenancy.py) is
not the same claim as it actually gating the real routes.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.auth import Principal
from core.tenancy import TenantConfig

client = TestClient(app)

LOW_RISK = ("LOW", {"semantic_score": 0.1, "source": "mock"})


def authenticated_as(key_id, tenant="acme"):
    return patch(
        "api.main.resolve_principal",
        return_value=Principal(
            capability="GENERAL", tenant=tenant, key_id=key_id,
            authenticated=True, reason="test",
        ),
    )


def tenant_resolves_to(config: TenantConfig):
    return patch("api.main.resolve_tenant", return_value=config)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """These tests are about suspension/SLA, not the limiter itself."""
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


# ---------------------------------------------------------------------------
# Suspension
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_suspended_tenant_is_rejected(_mock):
    suspended = TenantConfig(tenant_id="acme", status="suspended")
    with authenticated_as("k1"), tenant_resolves_to(suspended):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_suspension_is_checked_before_the_expensive_work(mock_assess):
    """The whole point: a rejected request must never reach assess_risk."""
    suspended = TenantConfig(tenant_id="acme", status="suspended")
    with authenticated_as("k1"), tenant_resolves_to(suspended):
        client.post("/api/v1/assess", json={"prompt": "hello"})

    assert mock_assess.call_count == 0


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_active_tenant_is_served_normally(_mock):
    active = TenantConfig(tenant_id="acme", status="active")
    with authenticated_as("k1"), tenant_resolves_to(active):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 200


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_unconfigured_tenant_is_unaffected(_mock):
    """No tenants.json entry at all -> DEFAULT_TENANT -> served normally.
    Regression guard: adding tenancy must not break existing deployments
    that have never configured a tenant."""
    response = client.post("/api/v1/assess", json={"prompt": "hello"})
    assert response.status_code == 200


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_output_endpoint_also_enforces_suspension(_mock):
    """The same gap §3.4 closed for auth/rate-limiting (assess_output had no
    controls at all) must not reopen for tenancy."""
    suspended = TenantConfig(tenant_id="acme", status="suspended")
    with authenticated_as("k1"), tenant_resolves_to(suspended), \
         patch("core.output_guardrails.assess_output", return_value=("ALLOW", {})):
        response = client.post("/api/v1/assess_output", json={"response_text": "hi"})

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# SLA -> rate limit override
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_tenant_rate_limit_overrides_the_tier_default(_mock, monkeypatch):
    """A tenant's configured SLA must actually change the limit applied, not
    just be recorded and ignored — the failure mode this module exists to
    fix for Principal.tenant itself."""
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_AUTHENTICATED_RPM", 600.0)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_BURST_SECONDS", 2.0)
    # Tenant SLA is far tighter than the tier default: 30/min = 0.5/sec,
    # burst 2s => capacity 1 token.
    tight = TenantConfig(tenant_id="acme", status="active", rate_limit_rpm=30.0)

    with authenticated_as("k1"), tenant_resolves_to(tight):
        first = client.post("/api/v1/assess", json={"prompt": "hello"})
        second = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert first.status_code == 200
    assert second.status_code == 429, (
        "tenant SLA of 30/min should have limited far sooner than the "
        "600/min tier default"
    )


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_no_override_falls_back_to_tier_default(_mock, monkeypatch):
    """rate_limit_rpm=None must mean 'use the tier default', not 'no limit'."""
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_AUTHENTICATED_RPM", 2.0)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_BURST_SECONDS", 1.0)
    no_override = TenantConfig(tenant_id="acme", status="active", rate_limit_rpm=None)

    with authenticated_as("k1"), tenant_resolves_to(no_override):
        codes = [
            client.post("/api/v1/assess", json={"prompt": "hello"}).status_code
            for _ in range(4)
        ]

    assert 429 in codes, "tier default (2/min) should still apply with no SLA override"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_tenant_override_never_applies_to_anonymous_callers(_mock, monkeypatch):
    """
    An anonymous caller has no verified tenant, so nothing it claims can
    carry an SLA override — otherwise an unauthenticated request could name
    a generous tenant_id and inherit its rate limit.
    """
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ANONYMOUS_RPM", 2.0)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_BURST_SECONDS", 1.0)
    generous = TenantConfig(tenant_id="acme", status="active", rate_limit_rpm=10_000.0)

    with tenant_resolves_to(generous):  # NOT authenticated
        codes = [
            client.post("/api/v1/assess", json={"prompt": "hello"}).status_code
            for _ in range(4)
        ]

    assert 429 in codes, "anonymous traffic must not inherit a tenant's SLA"


# ---------------------------------------------------------------------------
# Audit / metrics visibility
# ---------------------------------------------------------------------------

def test_tenant_reaches_the_audit_record():
    """Without this, tenancy is a field that exists and nothing reads —
    exactly the state this module was built to end."""
    import logging

    records = []
    audit_logger = logging.getLogger("gatekeeper.audit")

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    audit_logger.addHandler(handler)
    try:
        active = TenantConfig(tenant_id="acme", status="active")
        with authenticated_as("k1", tenant="acme"), tenant_resolves_to(active), \
             patch("api.main.assess_risk", return_value=LOW_RISK), \
             patch("api.main.settings.RATE_LIMIT_ENABLED", False):
            client.post("/api/v1/assess", json={"prompt": "hello"})
    finally:
        audit_logger.removeHandler(handler)

    assert records, "no audit record was emitted"
    assert getattr(records[-1], "tenant") == "acme"
