"""
Systematic auth-state x capability x endpoint matrix.

api/main.py has ~20 test files covering individual endpoints, but none of
them enumerate the full cross product in one place: for every endpoint that
requires a specific capability tier, or authentication at all, does it
actually enforce that boundary the same way every other endpoint does?

This file drives the REAL `app` object with FastAPI's TestClient and REAL
auth/tenant resolution -- real API keys registered through the same
KeyStore-on-a-tmp-file pattern tests/test_whoami_endpoint.py and
tests/test_policy_editor_endpoint.py already use, and a real suspended
tenant registered through core.tenancy.TenantStore the same way
tests/test_tenant_enforcement.py exercises it (there via mocking
resolve_tenant directly; here via a real on-disk tenant store, since this
file's job is to prove the real wiring end to end).

Two endpoint shapes exist in api/main.py, confirmed by reading every route:

1. "auth-required" endpoints (whoami always; assess/assess_output/
   gateway_chat/tools_call/settings.privacy/settings.protection/activity/
   activity.trace/review.status only when AUTH_MODE="required") check
   `principal.authenticated` FIRST and return 401 (with WWW-Authenticate)
   before anything else runs.

2. "INTERNAL-only" endpoints (gateway/providers, tools list, cache/flush,
   logs, benchmarks, the policy editor routes, review list, review resolve)
   check `principal.capability != CAPABILITY_INTERNAL` directly, with NO
   preceding authentication check. An anonymous caller resolves to GENERAL
   (core/auth.py's documented fail-open-but-unprivileged default), so a
   missing credential on these endpoints is a 403 ("INTERNAL capability
   required"), never a 401 -- there is no separate "you must log in" step
   to fail first. That is real, intentional behaviour (not a gap this file
   is inventing), and is asserted explicitly below rather than skipped.
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import auth as auth_mod
from core import review_queue as review_queue_mod
from core import tenancy as tenancy_mod
from core.auth import KeyStore, generate_key, hash_key
from core.review_queue import ReviewQueue
from core.tenancy import TenantStore

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key_store(tmp_path, monkeypatch):
    """Same pattern as test_whoami_endpoint.py / test_policy_editor_endpoint.py:
    real KeyStore backed by a tmp JSON file, swapped into core.auth's module
    singleton so resolve_principal() resolves against it for real."""
    path = tmp_path / "api_keys.json"
    store = {}

    def issue(capability="GENERAL", tenant="default", key_id="test-key"):
        plaintext = generate_key()
        store[hash_key(plaintext)] = {"capability": capability, "tenant": tenant, "key_id": key_id}
        path.write_text(json.dumps(store), encoding="utf-8")
        monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
        return plaintext

    return issue


@pytest.fixture
def suspended_tenant(tmp_path, monkeypatch):
    """A REAL suspended tenant, resolved through core.tenancy's real
    TenantStore/resolve_tenant (not mocked) -- registers 'suspended-co' as
    status=suspended so a key issued under that tenant hits the real
    suspension check every gated endpoint shares via
    _reject_suspended_and_rate_limit / the inline equivalent."""
    path = tmp_path / "tenants.json"
    path.write_text(
        json.dumps({"suspended-co": {"display_name": "Suspended Co", "status": "suspended"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))


@pytest.fixture
def review_queue(tmp_path, monkeypatch):
    """Real ReviewQueue backed by a tmp file, so review-endpoint tests
    exercise get_review/enqueue_review/resolve_review for real."""
    path = tmp_path / "review_queue.json"
    q = ReviewQueue(str(path))
    monkeypatch.setattr(review_queue_mod, "_queue", q)
    return q


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """This file is about auth/capability/tenant-suspension boundaries, not
    the rate limiter -- disable it so a large number of requests across many
    tests in one process can't spuriously 429."""
    monkeypatch.setattr("core.config.settings.RATE_LIMIT_ENABLED", False)


@pytest.fixture
def auth_required(monkeypatch):
    """Flips AUTH_MODE to 'required' for the endpoints that branch on
    auth_required() -- default is 'optional' (core/config.py), under which
    those endpoints serve anonymous callers instead of 401ing them."""
    monkeypatch.setattr("core.config.settings.AUTH_MODE", "required")


def auth_header(key):
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------------------
# Group 1: "auth-required" endpoints -- 401 (with WWW-Authenticate) when
# AUTH_MODE=required and no/garbage credential is presented.
# ---------------------------------------------------------------------------

AUTH_REQUIRED_GET_ENDPOINTS = [
    "/api/v1/settings/privacy",
    "/api/v1/settings/protection",
    "/api/v1/activity",
    "/api/v1/activity/trace/some-request-id",
    "/api/v1/review/some-review-id",
]


@pytest.mark.parametrize("path", AUTH_REQUIRED_GET_ENDPOINTS)
def test_no_credential_is_401_when_auth_required(path, auth_required):
    response = client.get(path)
    assert response.status_code == 401, f"{path}: expected 401, got {response.status_code}: {response.text}"
    assert response.headers.get("www-authenticate") == "Bearer"


@pytest.mark.parametrize("path", AUTH_REQUIRED_GET_ENDPOINTS)
@pytest.mark.parametrize("bad_header", ["not-bearer-shaped", "", "Bearer", "Bearer "])
def test_malformed_authorization_header_is_clean_401(path, bad_header, auth_required):
    response = client.get(path, headers={"Authorization": bad_header})
    assert response.status_code == 401, f"{path} / {bad_header!r}: got {response.status_code}: {response.text}"


def test_assess_no_credential_is_401_when_auth_required(auth_required):
    response = client.post("/api/v1/assess", json={"prompt": "hello"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_assess_output_no_credential_is_401_when_auth_required(auth_required):
    response = client.post("/api/v1/assess_output", json={"response_text": "hello"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_gateway_chat_no_credential_is_401_when_auth_required(auth_required):
    response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_tools_call_no_credential_is_401_when_auth_required(auth_required):
    response = client.post("/api/v1/tools/call", json={"name": "does.not.exist", "arguments": {}})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


@pytest.mark.parametrize("bad_header", ["not-bearer-shaped", "", "garbage-token-no-scheme"])
def test_assess_malformed_header_is_clean_401(bad_header, auth_required):
    response = client.post("/api/v1/assess", json={"prompt": "hello"}, headers={"Authorization": bad_header})
    assert response.status_code == 401


@pytest.mark.parametrize("bad_header", ["not-bearer-shaped", ""])
def test_gateway_chat_malformed_header_is_clean_401(bad_header, auth_required):
    response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"}, headers={"Authorization": bad_header})
    assert response.status_code == 401


# whoami ALWAYS requires a valid credential, independent of AUTH_MODE.

def test_whoami_no_credential_is_401_regardless_of_auth_mode():
    response = client.get("/api/v1/whoami")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


@pytest.mark.parametrize("bad_header", ["not-bearer-shaped", "", "Bearer", "Bearer   "])
def test_whoami_malformed_header_is_clean_401(bad_header):
    response = client.get("/api/v1/whoami", headers={"Authorization": bad_header})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Group 2: INTERNAL-only endpoints -- GENERAL -> 403, INTERNAL -> success,
# and (documented real behaviour) no credential -> 403, not 401, because
# these endpoints check capability directly without a preceding
# authentication gate.
# ---------------------------------------------------------------------------

INTERNAL_ONLY_GET_ENDPOINTS = [
    "/api/v1/gateway/providers",
    "/api/v1/tools",
    "/api/v1/logs",
    "/api/v1/benchmarks",
    "/api/v1/policy",
    "/api/v1/review",
]


@pytest.mark.parametrize("path", INTERNAL_ONLY_GET_ENDPOINTS)
def test_general_key_on_internal_endpoint_is_403(path, key_store):
    key = key_store(capability="GENERAL")
    response = client.get(path, headers=auth_header(key))
    assert response.status_code == 403, f"{path}: expected 403, got {response.status_code}: {response.text}"


@pytest.mark.parametrize("path", INTERNAL_ONLY_GET_ENDPOINTS)
def test_internal_key_on_internal_endpoint_is_not_403(path, key_store):
    key = key_store(capability="INTERNAL")
    response = client.get(path, headers=auth_header(key))
    assert response.status_code != 403, f"{path}: INTERNAL key got 403: {response.text}"
    assert 200 <= response.status_code < 300, f"{path}: expected 2xx, got {response.status_code}: {response.text}"


@pytest.mark.parametrize("path", INTERNAL_ONLY_GET_ENDPOINTS)
def test_no_credential_on_internal_endpoint_is_403_not_401(path):
    """Documents real behaviour: these routes check
    `principal.capability != CAPABILITY_INTERNAL` with no preceding
    authentication check, so an anonymous caller (resolved to GENERAL) is
    rejected as under-privileged, not unauthenticated -- true regardless of
    AUTH_MODE, since none of these branch on auth_required() at all."""
    response = client.get(path)
    assert response.status_code == 403, f"{path}: expected 403, got {response.status_code}: {response.text}"


def test_cache_flush_general_key_is_403(key_store):
    key = key_store(capability="GENERAL")
    response = client.post("/api/v1/cache/flush", headers=auth_header(key))
    assert response.status_code == 403


def test_cache_flush_internal_key_succeeds(key_store):
    key = key_store(capability="INTERNAL")
    response = client.post("/api/v1/cache/flush", headers=auth_header(key))
    assert response.status_code == 200
    assert response.json() == {"status": "success"}


def test_cache_flush_no_credential_is_403_not_401():
    response = client.post("/api/v1/cache/flush")
    assert response.status_code == 403


# --- Policy editor sub-routes (POST) --------------------------------------

VALID_POLICY = {
    "default_action": "BLOCK",
    "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK", "LOW": "ALLOW"}}}},
}


@pytest.fixture(autouse=True)
def _isolated_policy_files(tmp_path, monkeypatch):
    """Isolates the live policy file/versions dir the same way
    tests/test_policy_editor_endpoint.py does, so policy/validate and
    policy/deploy in this file never touch the real project's policy_rules
    file, and so 200s asserted below are real, not accidental."""
    from core import policy as policy_mod
    from core.policy import PolicyStore

    live = tmp_path / "policy_rules.json"
    live.write_text(json.dumps(VALID_POLICY), encoding="utf-8")
    versions_dir = tmp_path / "policy_versions"
    monkeypatch.setattr("core.config.settings.POLICY_RULES_FILE", str(live))
    monkeypatch.setattr("core.config.settings.POLICY_VERSIONS_DIR", str(versions_dir))
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(live)))


def test_policy_validate_general_key_is_403(key_store):
    key = key_store(capability="GENERAL")
    response = client.post("/api/v1/policy/validate", json={"content": "{}"}, headers=auth_header(key))
    assert response.status_code == 403


def test_policy_validate_internal_key_succeeds(key_store):
    key = key_store(capability="INTERNAL")
    response = client.post(
        "/api/v1/policy/validate", json={"content": json.dumps(VALID_POLICY)}, headers=auth_header(key)
    )
    assert response.status_code == 200
    assert response.json() == {"valid": True, "errors": []}


def test_policy_validate_no_credential_is_403_not_401():
    response = client.post("/api/v1/policy/validate", json={"content": "{}"})
    assert response.status_code == 403


def test_policy_deploy_general_key_is_403(key_store):
    key = key_store(capability="GENERAL")
    response = client.post(
        "/api/v1/policy/deploy", json={"content": json.dumps(VALID_POLICY)}, headers=auth_header(key)
    )
    assert response.status_code == 403


def test_policy_deploy_internal_key_succeeds(key_store):
    key = key_store(capability="INTERNAL")
    response = client.post(
        "/api/v1/policy/deploy", json={"content": json.dumps(VALID_POLICY)}, headers=auth_header(key)
    )
    assert response.status_code == 200
    assert response.json()["deployed"] is True


def test_policy_rollback_general_key_is_403(key_store):
    key = key_store(capability="GENERAL")
    response = client.post("/api/v1/policy/rollback", json={"version": "does-not-exist.json"}, headers=auth_header(key))
    assert response.status_code == 403


def test_policy_rollback_internal_key_is_not_403(key_store):
    """INTERNAL clears the capability gate even though the named version
    doesn't exist -- proves the 403 in the previous test is capability, not
    incidental, since this hits the same nonexistent-version path and gets
    404 (business-logic error) rather than 403 (authorization error)."""
    key = key_store(capability="INTERNAL")
    response = client.post("/api/v1/policy/rollback", json={"version": "does-not-exist.json"}, headers=auth_header(key))
    assert response.status_code == 404
    assert response.status_code != 403


# --- Review list / resolve (INTERNAL-only) --------------------------------

def test_review_list_general_key_is_403(key_store, review_queue):
    key = key_store(capability="GENERAL")
    response = client.get("/api/v1/review", headers=auth_header(key))
    assert response.status_code == 403


def test_review_list_internal_key_succeeds(key_store, review_queue):
    review_queue.enqueue(
        reason="test", capability="GENERAL", risk="HIGH", tenant="default",
        prompt_hash="deadbeef", request_id="req-1",
    )
    key = key_store(capability="INTERNAL")
    response = client.get("/api/v1/review", headers=auth_header(key))
    assert response.status_code == 200
    assert len(response.json()["pending"]) == 1


def test_review_resolve_general_key_is_403(key_store, review_queue):
    record = review_queue.enqueue(
        reason="test", capability="GENERAL", risk="HIGH", tenant="default",
        prompt_hash="deadbeef", request_id="req-2",
    )
    key = key_store(capability="GENERAL")
    response = client.post(
        f"/api/v1/review/{record.review_id}/resolve", json={"outcome": "APPROVED"}, headers=auth_header(key)
    )
    assert response.status_code == 403


def test_review_resolve_internal_key_succeeds(key_store, review_queue):
    record = review_queue.enqueue(
        reason="test", capability="GENERAL", risk="HIGH", tenant="default",
        prompt_hash="deadbeef", request_id="req-3",
    )
    key = key_store(capability="INTERNAL")
    response = client.post(
        f"/api/v1/review/{record.review_id}/resolve", json={"outcome": "APPROVED"}, headers=auth_header(key)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "APPROVED"
    assert body["final_decision"] == "ALLOW"


def test_review_resolve_no_credential_is_403_not_401(review_queue):
    record = review_queue.enqueue(
        reason="test", capability="GENERAL", risk="HIGH", tenant="default",
        prompt_hash="deadbeef", request_id="req-4",
    )
    response = client.post(f"/api/v1/review/{record.review_id}/resolve", json={"outcome": "APPROVED"})
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Group 3: a suspended tenant's otherwise-valid key -> 403 tenant-suspended,
# against a REAL suspended tenant resolved through core.tenancy.
# ---------------------------------------------------------------------------

def test_whoami_suspended_tenant_is_403(key_store, suspended_tenant):
    key = key_store(capability="GENERAL", tenant="suspended-co")
    response = client.get("/api/v1/whoami", headers=auth_header(key))
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_settings_privacy_suspended_tenant_is_403(key_store, suspended_tenant):
    key = key_store(capability="GENERAL", tenant="suspended-co")
    response = client.get("/api/v1/settings/privacy", headers=auth_header(key))
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_activity_suspended_tenant_is_403(key_store, suspended_tenant):
    key = key_store(capability="GENERAL", tenant="suspended-co")
    response = client.get("/api/v1/activity", headers=auth_header(key))
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_assess_suspended_tenant_is_403(key_store, suspended_tenant):
    key = key_store(capability="GENERAL", tenant="suspended-co")
    response = client.post("/api/v1/assess", json={"prompt": "hello"}, headers=auth_header(key))
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_assess_output_suspended_tenant_is_403(key_store, suspended_tenant):
    key = key_store(capability="GENERAL", tenant="suspended-co")
    response = client.post(
        "/api/v1/assess_output", json={"response_text": "hello"}, headers=auth_header(key)
    )
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_tools_call_suspended_tenant_is_403(key_store, suspended_tenant):
    key = key_store(capability="GENERAL", tenant="suspended-co")
    response = client.post(
        "/api/v1/tools/call", json={"name": "does.not.exist", "arguments": {}}, headers=auth_header(key)
    )
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_review_status_suspended_tenant_is_403(key_store, suspended_tenant, review_queue):
    key = key_store(capability="GENERAL", tenant="suspended-co")
    response = client.get("/api/v1/review/some-review-id", headers=auth_header(key))
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_internal_key_suspended_tenant_still_rejected_on_internal_endpoint(key_store, suspended_tenant):
    """Suspension is checked AFTER the capability gate on INTERNAL-only
    endpoints (see api/main.py's `_require_internal` / inline equivalents),
    so even an INTERNAL key under a suspended tenant is rejected -- capability
    alone does not exempt a caller from their tenant's suspension."""
    key = key_store(capability="INTERNAL", tenant="suspended-co")
    response = client.get("/api/v1/logs", headers=auth_header(key))
    assert response.status_code == 403
    assert "suspended" in response.json()["detail"].lower()


def test_active_tenant_key_not_rejected_as_suspended(key_store, suspended_tenant):
    """Control: an otherwise-identical key under an unrelated (unconfigured
    -> DEFAULT_TENANT, active) tenant must not be caught by the same check."""
    key = key_store(capability="GENERAL", tenant="default")
    response = client.get("/api/v1/whoami", headers=auth_header(key))
    assert response.status_code == 200
