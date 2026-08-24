"""
Phase 8 hardening: rate limiting and tenant-suspension enforcement,
extended to every endpoint added or exposed over HTTP in Phase 7.

Before this, whoami / activity / trace / logs / gateway providers / tool
catalogue / benchmarks / policy editor / the review endpoints / cache-flush
had NEITHER check -- only the original four endpoints (assess,
assess_output, gateway/chat, tools/call) enforced suspension and rate
limiting. A suspended tenant could poll any of these indefinitely, and
several do real per-call work (a bounded audit-log scan, a policy-file
write+validate+delete) that benefits from the same throttling the rest of
the API already has.

Same API-level testing approach as test_request_limits.py and
test_tenant_enforcement.py: these controls only matter if they fire on
the real routes, not in isolation.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.auth import Principal
from core.tenancy import TenantConfig

client = TestClient(app)


def authenticated_as(key_id, tenant="acme", capability="GENERAL"):
    return patch(
        "api.main.resolve_principal",
        return_value=Principal(
            capability=capability, tenant=tenant, key_id=key_id,
            authenticated=True, reason="test",
        ),
    )


def tenant_resolves_to(config: TenantConfig):
    return patch("api.main.resolve_tenant", return_value=config)


@pytest.fixture
def fast_limit(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_AUTHENTICATED_RPM", 600.0)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_BURST_SECONDS", 0.05)  # ~1-token bucket at 600rpm


@pytest.fixture(autouse=True)
def _no_rate_limit_by_default(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


@pytest.fixture(autouse=True)
def _isolated_audit_log(tmp_path, monkeypatch):
    monkeypatch.setattr("core.activity.settings.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))


# --- Suspension: representative endpoints across every auth tier ---------------

@pytest.mark.parametrize("method,url,json_body", [
    ("get", "/api/v1/whoami", None),
    ("get", "/api/v1/activity", None),
    ("get", "/api/v1/activity/trace/some-id", None),
    ("get", "/api/v1/settings/privacy", None),
    ("get", "/api/v1/settings/protection", None),
])
def test_suspended_tenant_rejected_on_general_endpoints(method, url, json_body):
    suspended = TenantConfig(tenant_id="acme", status="suspended")
    kwargs = {"json": json_body} if method == "post" else {}
    with authenticated_as("k1"), tenant_resolves_to(suspended):
        response = getattr(client, method)(url, **kwargs)
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


@pytest.mark.parametrize("method,url,json_body", [
    ("get", "/api/v1/logs", None),
    ("get", "/api/v1/benchmarks", None),
    ("get", "/api/v1/gateway/providers", None),
    ("get", "/api/v1/tools", None),
    ("get", "/api/v1/policy", None),
    ("get", "/api/v1/review", None),
    ("post", "/api/v1/cache/flush", None),
])
def test_suspended_tenant_rejected_on_internal_endpoints(method, url, json_body):
    suspended = TenantConfig(tenant_id="acme", status="suspended")
    kwargs = {"json": json_body} if method == "post" else {}
    with authenticated_as("k1", capability="INTERNAL"), tenant_resolves_to(suspended):
        response = getattr(client, method)(url, **kwargs)
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_suspended_tenant_rejected_on_policy_validate():
    suspended = TenantConfig(tenant_id="acme", status="suspended")
    with authenticated_as("k1", capability="INTERNAL"), tenant_resolves_to(suspended):
        response = client.post("/api/v1/policy/validate", json={"content": "{}"})
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_active_tenant_still_served_normally_on_a_new_endpoint(_isolated_audit_log):
    active = TenantConfig(tenant_id="acme", status="active")
    with authenticated_as("k1"), tenant_resolves_to(active):
        response = client.get("/api/v1/activity")
    assert response.status_code == 200


# --- Rate limiting: representative endpoints ------------------------------------

def test_activity_feed_is_rate_limited(fast_limit):
    with authenticated_as("k1"):
        codes = [client.get("/api/v1/activity").status_code for _ in range(6)]
    assert 429 in codes, "sustained overage on a Phase 7 read endpoint must be rejected"


def test_whoami_is_rate_limited(fast_limit):
    with authenticated_as("k1"):
        codes = [client.get("/api/v1/whoami").status_code for _ in range(6)]
    assert 429 in codes


def test_policy_view_is_rate_limited(fast_limit):
    with authenticated_as("k1", capability="INTERNAL"):
        codes = [client.get("/api/v1/policy").status_code for _ in range(6)]
    assert 429 in codes


def test_rate_limit_429_includes_retry_after(fast_limit):
    with authenticated_as("k1"):
        for _ in range(10):
            response = client.get("/api/v1/activity")
            if response.status_code == 429:
                break
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1


def test_unauthenticated_whoami_still_gets_401_not_a_rate_limit_bypass_of_the_check_order():
    """The 401 (invalid credential) check must still fire -- adding rate
    limiting must not have reordered it past the credential check."""
    response = client.get("/api/v1/whoami")
    assert response.status_code == 401


# --- Real content-size bound on the policy editor (Phase 8 hardening) ---------

def test_policy_content_oversized_body_is_rejected():
    huge = "x" * 1_000_001
    with authenticated_as("k1", capability="INTERNAL"):
        response = client.post("/api/v1/policy/validate", json={"content": huge})
    assert response.status_code == 422


def test_policy_rollback_version_oversized_is_rejected():
    with authenticated_as("k1", capability="INTERNAL"):
        response = client.post("/api/v1/policy/rollback", json={"version": "x" * 256})
    assert response.status_code == 422
