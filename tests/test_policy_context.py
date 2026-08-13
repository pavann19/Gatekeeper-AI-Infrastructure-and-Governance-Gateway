"""
Endpoint-level proof that Policy Context is load-bearing on the real API,
not just in the module under test. Same reasoning as
tests/test_tenant_enforcement.py: resolve_policy_set() working in isolation
is not the same claim as it actually deciding what /assess returns.
"""
import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app
from core import policy as policy_mod
from core.auth import Principal
from core.policy import PolicyStore

client = TestClient(app)

MEDIUM_RISK = ("MEDIUM", {"semantic_score": 0.2, "source": "mock"})


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


@pytest.fixture
def two_tenant_policies(tmp_path, monkeypatch):
    """default RESTRICTs a MEDIUM GENERAL risk; strict-acme BLOCKs it."""
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {
            "default": {"policies": {"GENERAL": {"MEDIUM": "RESTRICT"}}},
            "strict-acme": {"policies": {"GENERAL": {"MEDIUM": "BLOCK"}}},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))


def authenticated_as(key_id, tenant):
    return patch(
        "api.main.resolve_principal",
        return_value=Principal(
            capability="GENERAL", tenant=tenant, key_id=key_id,
            authenticated=True, reason="test",
        ),
    )


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_same_risk_different_decision_by_tenant(_mock, two_tenant_policies):
    """THE END-TO-END PROOF: identical risk assessment, different tenant,
    different HTTP-level decision — not just a different internal value."""
    with authenticated_as("k1", "default"):
        default_resp = client.post("/api/v1/assess", json={"prompt": "hello"})
    with authenticated_as("k2", "strict-acme"):
        strict_resp = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert default_resp.json()["decision"] == "RESTRICT"
    assert strict_resp.json()["decision"] == "BLOCK"


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_policy_uses_the_server_resolved_tenant_not_a_client_claim(_mock, two_tenant_policies):
    """
    THE SECURITY PROPERTY: tenant comes from the verified Principal, exactly
    like capability does (core/auth.py's whole reason for existing) — a
    request body cannot claim a looser tenant to get a softer decision.
    AssessRequest has no tenant field at all, so this is really asserting
    that a stray field is rejected outright (extra="forbid", set in §3.4).
    """
    with authenticated_as("k1", "default"):
        response = client.post(
            "/api/v1/assess",
            json={"prompt": "hello", "tenant": "strict-acme"},
        )
    assert response.status_code == 422, "a client-supplied tenant field must be rejected, not honoured"


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_unconfigured_tenant_gets_default_policy_on_the_real_endpoint(_mock, two_tenant_policies):
    with authenticated_as("k3", "some-other-tenant"):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})
    assert response.json()["decision"] == "RESTRICT"


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_policy_reason_is_recorded_in_response_details(_mock, two_tenant_policies):
    with authenticated_as("k2", "strict-acme"):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})
    assert "strict-acme" in response.json()["details"]["policy_reason"]
